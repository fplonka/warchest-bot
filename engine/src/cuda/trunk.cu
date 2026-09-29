#define SLD (KC + 8)
#define STAGES 3
#define TROWS (TB * N_HEXES)
#define TM ((TROWS + 15) / 16 * 16)
#define TFN (TRUNK_C / 8 / TWARPS)
#define DFN (DN / 8 / DWARPS)
static_assert(TFN % 4 == 0 && DFN % 4 == 0 && TRUNK_C == 8 * TFN * TWARPS && DN == 8 * DFN * DWARPS,
              "a warp owns whole column tiles and a lane whole 8-column words");
static_assert(TRUNK_C % KC == 0 && HK % KC == 0 && D % KC == 0, "every reduction is whole chunks");
static_assert(TWARPS * TM <= STAGES * TM * SLD / 4, "the row statistics fit in the staging buffers");

struct H16 {
    unsigned r[2];
};

struct F32 {
    float r[4];
};

__device__ __forceinline__ void mma(H16& d, const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(d.r[0]), "+r"(d.r[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma(F32& d, const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d.r[0]), "+f"(d.r[1]), "+f"(d.r[2]), "+f"(d.r[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ float2 pair(const H16& d, int h) { return word2(d.r[h]); }

__device__ __forceinline__ float2 pair(const F32& d, int h) { return make_float2(d.r[2 * h], d.r[2 * h + 1]); }

template <int FN>
__device__ __forceinline__ int column(int nt0, int n) {
    return 8 * (nt0 + (threadIdx.x >> 5) * FN) + 32 * (n / 4) + 8 * (threadIdx.x & 3) + 2 * (n % 4);
}

__device__ __forceinline__ uint4 quad(const H16 (&acc)[TFN], int q, int hf) {
    return make_uint4(acc[4 * q].r[hf], acc[4 * q + 1].r[hf], acc[4 * q + 2].r[hf], acc[4 * q + 3].r[hf]);
}

__device__ __forceinline__ void frag_a16(const __half* a, int row, int k, int lane, unsigned (&f)[4]) {
    const __half* p = a + (row + (lane & 7) + 8 * ((lane >> 3) & 1)) * SLD + k + 8 * (lane >> 4);
    unsigned addr = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3]) : "r"(addr));
}

__device__ __forceinline__ void frag_b16(const unsigned* __restrict__ w, int kt, int nt, int ntiles, int lane,
                                         unsigned (&f)[2]) {
    const uint2 v = __ldg((const uint2*)(w + (((size_t)kt * ntiles + nt) * 32 + lane) * 2));
    f[0] = v.x;
    f[1] = v.y;
}

__device__ __forceinline__ void cp_async16(void* s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((unsigned)__cvta_generic_to_shared(s)), "l"(g));
}

template <int MT, int WARPS>
__device__ __forceinline__ void stage(__half* s, const __half* a, int lda, int rows, int row0, int k0) {
    for (int i = threadIdx.x; i < MT * KC / 8; i += 32 * WARPS) {
        const int r = i / (KC / 8), c = i % (KC / 8);
        cp_async16(s + r * SLD + 8 * c, a + (size_t)min(row0 + r, rows - 1) * lda + k0 + 8 * c);
    }
}

__device__ __forceinline__ void neighbours(const __half* sc, __half* sn, const unsigned char* __restrict__ near) {
    for (int i = threadIdx.x; i < TM * KC / 8; i += 32 * TWARPS) {
        const int r = i / (KC / 8), c = 8 * (i % (KC / 8));
        __half2 f[4] = {};
        if (r < TROWS) {
            const int base = r / N_HEXES * N_HEXES;
#pragma unroll
            for (int j = 0; j < 6; ++j) {
                const int o = __ldg(near + (r - base) * 6 + j);
                if (o >= N_HEXES) continue;
                const uint4 v = *(const uint4*)(sc + (base + o) * SLD + c);
                const __half2* u = (const __half2*)&v;
#pragma unroll
                for (int q = 0; q < 4; ++q) f[q] = __hadd2(f[q], u[q]);
            }
        }
        *(uint4*)(sn + r * SLD + c) = *(const uint4*)f;
    }
}

template <int MT, int FN, int WARPS, int PARTS, typename Frag>
__device__ __forceinline__ void gemm(const __half* a, int lda, int rows, int row0, int k,
                                     const unsigned* __restrict__ w, int ntiles, int nt0, __half* s,
                                     const unsigned char* __restrict__ near, Frag (&acc)[MT / 16][FN]) {
    constexpr int KS = KC / 16;
    const int lane = threadIdx.x & 31, nt = nt0 + (threadIdx.x >> 5) * FN, chunks = k / KC;
    const int steps = chunks * PARTS * KS;
    const auto kt_of = [&](int i) {
        const int q = i % (PARTS * KS);
        return q / KS * (k / 16) + i / (PARTS * KS) * KS + q % KS;
    };
    __half* sn = s + STAGES * MT * SLD;
#pragma unroll
    for (int m = 0; m < MT / 16; ++m)
#pragma unroll
        for (int n = 0; n < FN; ++n) acc[m][n] = Frag{};
#pragma unroll
    for (int c = 0; c < STAGES - 1; ++c) {
        if (c < chunks) stage<MT, WARPS>(s + c * MT * SLD, a, lda, rows, row0, c * KC);
        asm volatile("cp.async.commit_group;");
    }
    if constexpr (PARTS == 2) {
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2));
        __syncthreads();
        neighbours(s, sn, near);
    }
    unsigned bf[FN][2];
#pragma unroll
    for (int n = 0; n < FN; ++n) frag_b16(w, 0, nt + n, ntiles, lane, bf[n]);
    for (int c = 0; c < chunks; ++c) {
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        const __half* sc = s + c % STAGES * MT * SLD;
        const __half* sd = sn + c % 2 * MT * SLD;
#pragma unroll
        for (int q = 0; q < PARTS * KS; ++q) {
            const int kt = kt_of(min(c * PARTS * KS + q + 1, steps - 1));
            unsigned bn[FN][2];
#pragma unroll
            for (int n = 0; n < FN; ++n) frag_b16(w, kt, nt + n, ntiles, lane, bn[n]);
#pragma unroll
            for (int m = 0; m < MT / 16; ++m) {
                unsigned af[4];
                frag_a16(q < KS ? sc : sd, 16 * m, 16 * (q % KS), lane, af);
#pragma unroll
                for (int n = 0; n < FN; ++n) mma(acc[m][n], af, bf[n]);
            }
            if (q == 0) {
                const int next = c + STAGES - 1;
                if (next < chunks) stage<MT, WARPS>(s + next % STAGES * MT * SLD, a, lda, rows, row0, next * KC);
                asm volatile("cp.async.commit_group;");
                if constexpr (PARTS == 2)
                    if (c + 1 < chunks) neighbours(s + (c + 1) % STAGES * MT * SLD, sn + (c + 1) % 2 * MT * SLD, near);
            }
#pragma unroll
            for (int n = 0; n < FN; ++n) {
                bf[n][0] = bn[n][0];
                bf[n][1] = bn[n][1];
            }
        }
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
}

template <typename Row>
__device__ __forceinline__ void add_rows(H16 (&acc)[TM / 16][TFN], const float* __restrict__ bias, Row row) {
    const int g = (threadIdx.x & 31) >> 2;
    float2 b[TFN];
#pragma unroll
    for (int n = 0; n < TFN; ++n) b[n] = *(const float2*)(bias + column<TFN>(0, n));
#pragma unroll
    for (int m = 0; m < TM / 16; ++m)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const __half* src = row(16 * m + g + 8 * hf);
#pragma unroll
            for (int q = 0; q < TFN / 4; ++q) {
                const uint4 v = *(const uint4*)(src + column<TFN>(0, 4 * q));
                const unsigned* u = (const unsigned*)&v;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int n = 4 * q + j;
                    const float2 y = pair(acc[m][n], hf), o = word2(u[j]);
                    acc[m][n].r[hf] = pack2(y.x + b[n].x + o.x, y.y + b[n].y + o.y);
                }
            }
        }
}

__device__ __forceinline__ void norm_rows(H16 (&acc)[TM / 16][TFN], float2* part, float2* stats,
                                          const float* __restrict__ gamma, const float* __restrict__ beta) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int m = 0; m < TM / 16; ++m)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            float sum = 0.0f, sq = 0.0f;
#pragma unroll
            for (int n = 0; n < TFN; ++n) {
                const float2 y = pair(acc[m][n], hf);
                sum += y.x + y.y;
                sq += y.x * y.x + y.y * y.y;
            }
#pragma unroll
            for (int o = 1; o < 4; o <<= 1) {
                sum += __shfl_xor_sync(0xffffffff, sum, o);
                sq += __shfl_xor_sync(0xffffffff, sq, o);
            }
            if (t == 0) part[warp * TM + 16 * m + g + 8 * hf] = make_float2(sum, sq);
        }
    __syncthreads();
    for (int r = threadIdx.x; r < TM; r += 32 * TWARPS) {
        float sum = 0.0f, sq = 0.0f;
#pragma unroll
        for (int k = 0; k < TWARPS; ++k) {
            sum += part[k * TM + r].x;
            sq += part[k * TM + r].y;
        }
        const float mean = sum / TRUNK_C;
        stats[r] = make_float2(mean, rsqrtf(fmaxf(sq / TRUNK_C - mean * mean, 0.0f) + 1e-5f));
    }
    float2 gm[TFN], bt[TFN];
#pragma unroll
    for (int n = 0; n < TFN; ++n) {
        gm[n] = *(const float2*)(gamma + column<TFN>(0, n));
        bt[n] = *(const float2*)(beta + column<TFN>(0, n));
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < TM / 16; ++m)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const float2 st = stats[16 * m + g + 8 * hf];
#pragma unroll
            for (int n = 0; n < TFN; ++n) {
                const float2 y = pair(acc[m][n], hf);
                acc[m][n].r[hf] = gelu_word((y.x - st.x) * st.y * gm[n].x + bt[n].x,
                                            (y.y - st.x) * st.y * gm[n].y + bt[n].y);
            }
        }
}

__device__ __forceinline__ void store_rows(const H16 (&acc)[TM / 16][TFN], __half* dst, int row0, int live) {
    const int g = (threadIdx.x & 31) >> 2;
#pragma unroll
    for (int m = 0; m < TM / 16; ++m)
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const int r = 16 * m + g + 8 * hf;
            if (r >= live) continue;
#pragma unroll
            for (int q = 0; q < TFN / 4; ++q)
                *(uint4*)(dst + (size_t)(row0 + r) * TRUNK_C + column<TFN>(0, 4 * q)) = quad(acc[m], q, hf);
        }
}

__device__ __forceinline__ void pool_rows(const H16 (&acc)[TM / 16][TFN], __half* pooled, int board0, int boards) {
    const int g = (threadIdx.x & 31) >> 2;
#pragma unroll
    for (int n = 0; n < TFN; ++n) {
        float sum[TB][2], top[TB][2];
#pragma unroll
        for (int b = 0; b < TB; ++b) sum[b][0] = sum[b][1] = 0.0f, top[b][0] = top[b][1] = neg_inf();
#pragma unroll
        for (int m = 0; m < TM / 16; ++m)
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                const int lo = 16 * m + 8 * hf;
                const int b = lo / N_HEXES == (lo + 7) / N_HEXES ? lo / N_HEXES : (lo + g) / N_HEXES;
                const float2 v = pair(acc[m][n], hf);
#pragma unroll
                for (int bb = 0; bb < TB; ++bb)
                    if (bb == b) {
                        sum[bb][0] += v.x;
                        sum[bb][1] += v.y;
                        top[bb][0] = fmaxf(top[bb][0], v.x);
                        top[bb][1] = fmaxf(top[bb][1], v.y);
                    }
            }
        const int col = column<TFN>(0, n);
#pragma unroll
        for (int bb = 0; bb < TB; ++bb) {
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int o = 4; o < 32; o <<= 1) {
                    sum[bb][j] += __shfl_xor_sync(0xffffffff, sum[bb][j], o);
                    top[bb][j] = fmaxf(top[bb][j], __shfl_xor_sync(0xffffffff, top[bb][j], o));
                }
            if (g == 0 && board0 + bb < boards) {
                __half* p = pooled + (size_t)(board0 + bb) * HK;
                *(unsigned*)(p + col) = pack2(sum[bb][0] / N_HEXES, sum[bb][1] / N_HEXES);
                *(unsigned*)(p + TRUNK_C + col) = pack2(top[bb][0], top[bb][1]);
            }
        }
    }
}

struct Stem {
    const float* tok;
    const unsigned char* ids;
    const float* w_hex;
    const float* b_hex;
    const float* pos;
    const float* w_glob;
    const float* gamma;
    const float* beta;
};

extern "C" __global__ void __launch_bounds__(TRUNK_C)
k_stem(const float* __restrict__ xpub, const Stem st, __half* x, __half* a, __half* tokens, __half* p) {
    __shared__ float pj[NTYPE][TRUNK_C];
    __shared__ float hexf[N_HEXES][HEX_CH];
    __shared__ float loose[LOOSE];
    __shared__ float2 part[TRUNK_C / 32][N_HEXES];
    const int b = blockIdx.x, c = threadIdx.x, lane = c & 31, warp = c >> 5;
    const float* xb = xpub + (size_t)b * PUBFEAT;
    for (int i = c; i < N_HEXES * HEX_CH; i += TRUNK_C) hexf[i / HEX_CH][i % HEX_CH] = xb[i];
    for (int i = c; i < HK - 2 * TRUNK_C; i += TRUNK_C) {
        const float v = i < LOOSE ? xb[OFF_LOOSE + i] : 0.0f;
        if (i < LOOSE) loose[i] = v;
        p[(size_t)b * HK + 2 * TRUNK_C + i] = __float2half(v);
    }
    float tp = 0.0f;
    for (int t = 0; t < NTYPE; ++t) {
        float v = st.tok[(size_t)st.ids[(size_t)b * ROW_BYTES + ROW_IDS + t] * TRUNK_C + c]
                + st.tok[(size_t)(N_UNITS + t / NSLOT) * TRUNK_C + c];
#pragma unroll
        for (int k = 0; k < PILE_COUNTS; ++k)
            v += xb[OFF_PILES + t * PILE_COUNTS + k] * st.tok[(size_t)(N_UNITS + 2 + k) * TRUNK_C + c];
        pj[t][c] = v;
        tokens[((size_t)b * NTYPE + t) * TRUNK_C + c] = __float2half(v);
        tp += gelu1(v);
    }
    __syncthreads();
    float base = st.b_hex[c] + tp / NTYPE;
    for (int k = 0; k < LOOSE; ++k) base += loose[k] * st.w_glob[(size_t)k * TRUNK_C + c];
    float v[N_HEXES];
#pragma unroll
    for (int h = 0; h < N_HEXES; ++h) v[h] = base + st.pos[(size_t)h * TRUNK_C + c];
    for (int f = 0; f < HEX_CH; ++f) {
        const float wv = f < HEX_FACTS ? st.w_hex[(size_t)f * TRUNK_C + c] : pj[f - HEX_FACTS][c];
#pragma unroll
        for (int h = 0; h < N_HEXES; ++h) v[h] += hexf[h][f] * wv;
    }
#pragma unroll
    for (int h = 0; h < N_HEXES; ++h) {
        const __half q = __float2half(v[h]);
        x[((size_t)b * N_HEXES + h) * TRUNK_C + c] = q;
        v[h] = __half2float(q);
        const float2 s = make_float2(warp_sum(v[h]), warp_sum(v[h] * v[h]));
        if (lane == 0) part[warp][h] = s;
    }
    __syncthreads();
    const float gm = st.gamma[c], bt = st.beta[c];
    float sum = 0.0f, top = neg_inf();
#pragma unroll
    for (int h = 0; h < N_HEXES; ++h) {
        float s = 0.0f, q = 0.0f;
#pragma unroll
        for (int k = 0; k < TRUNK_C / 32; ++k) {
            s += part[k][h].x;
            q += part[k][h].y;
        }
        const float mean = s / TRUNK_C, inv = rsqrtf(fmaxf(q / TRUNK_C - mean * mean, 0.0f) + 1e-5f);
        const __half o = __float2half(gelu1((v[h] - mean) * inv * gm + bt));
        a[((size_t)b * N_HEXES + h) * TRUNK_C + c] = o;
        sum += __half2float(o);
        top = fmaxf(top, __half2float(o));
    }
    p[(size_t)b * HK + c] = __float2half(sum / N_HEXES);
    p[(size_t)b * HK + TRUNK_C + c] = __float2half(top);
}

extern "C" __global__ void __launch_bounds__(32 * DWARPS)
k_dense(const __half* __restrict__ a, int lda, int k, const unsigned* __restrict__ w, int n,
        const float* __restrict__ bias, __half* out, int rows) {
    __shared__ __align__(16) __half s[STAGES * DM * SLD];
    const int row0 = blockIdx.x * DM, g = (threadIdx.x & 31) >> 2, nt0 = blockIdx.y * (DN / 8);
    F32 acc[DM / 16][DFN];
    gemm<DM, DFN, DWARPS, 1>(a, lda, rows, row0, k, w, n / 8, nt0, s, nullptr, acc);
#pragma unroll
    for (int m = 0; m < DM / 16; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = row0 + 16 * m + g + 8 * h;
            if (r >= rows) continue;
#pragma unroll
            for (int q = 0; q < DFN / 4; ++q) {
                unsigned o[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float2 v = pair(acc[m][4 * q + j], h), b = *(const float2*)(bias + column<DFN>(nt0, 4 * q + j));
                    o[j] = pack2(v.x + b.x, v.y + b.y);
                }
                *(uint4*)(out + (size_t)r * n + column<DFN>(nt0, 4 * q)) = make_uint4(o[0], o[1], o[2], o[3]);
            }
        }
}

extern "C" __global__ void __launch_bounds__(32 * TWARPS, 1)
k_mix(const __half* __restrict__ a, const unsigned* __restrict__ w, const unsigned char* __restrict__ near,
      const float* __restrict__ b1, const __half* __restrict__ g_pool, const float* __restrict__ gamma,
      const float* __restrict__ beta, __half* h, int rows) {
    __shared__ __align__(16) __half s[(STAGES + 2) * TM * SLD];
    __shared__ float2 stats[TM];
    const int row0 = blockIdx.x * TROWS;
    H16 acc[TM / 16][TFN];
    gemm<TM, TFN, TWARPS, 2>(a, TRUNK_C, rows, row0, TRUNK_C, w, TRUNK_C / 8, 0, s, near, acc);
    add_rows(acc, b1, [&](int r) { return g_pool + (size_t)(blockIdx.x * TB + min(r / N_HEXES, TB - 1)) * TRUNK_C; });
    norm_rows(acc, (float2*)s, stats, gamma, beta);
    store_rows(acc, h, row0, min(TROWS, rows - row0));
}

extern "C" __global__ void __launch_bounds__(32 * TWARPS, 1)
k_out(const __half* __restrict__ h, const unsigned* __restrict__ w, const float* __restrict__ b2,
      const float* __restrict__ gamma, const float* __restrict__ beta, __half* x, __half* a, __half* pooled,
      int rows) {
    __shared__ __align__(16) __half s[STAGES * TM * SLD];
    __shared__ float2 stats[TM];
    const int row0 = blockIdx.x * TROWS, live = min(TROWS, rows - row0);
    H16 acc[TM / 16][TFN];
    gemm<TM, TFN, TWARPS, 1>(h, TRUNK_C, rows, row0, TRUNK_C, w, TRUNK_C / 8, 0, s, nullptr, acc);
    add_rows(acc, b2, [&](int r) { return x + (size_t)(row0 + min(r, live - 1)) * TRUNK_C; });
    store_rows(acc, x, row0, live);
    norm_rows(acc, (float2*)s, stats, gamma, beta);
    store_rows(acc, a, row0, live);
    pool_rows(acc, pooled, blockIdx.x * TB, rows / N_HEXES);
}
