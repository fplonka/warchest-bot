
#define TB 3
#define TROWS (TB * N_HEXES)
#define TMT ((TROWS + 15) / 16)
#define TPROWS (TMT * 16)
#define TNT (TRUNK_C / 8)
#define TWN (TNT / TWARPS)
#define TLDS (TRUNK_C + 8)
#define TPACK (TRUNK_C * TRUNK_C / 2)
#define GLDS (2 * TRUNK_C + 8)
#define TQ (TRUNK_C / 64)
static_assert(TNT % TWARPS == 0, "k_trunk gives every warp the same number of column tiles");
static_assert(TRUNK_C % 64 == 0, "k_trunk row passes give every lane whole words");
static_assert(TWARPS % TB == 0, "k_trunk pools a board with a fixed set of warps");
static_assert(TRUNK_C / 16 <= TWARPS, "k_trunk projects the pooled rows with one warp a channel tile");

__device__ __forceinline__ void mma_h16(unsigned (&d)[2], const unsigned (&a)[4],
                                        const unsigned (&b)[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ float2 word2(unsigned u) {
    return make_float2(__half2float(__ushort_as_half((unsigned short)(u & 0xffffu))),
                       __half2float(__ushort_as_half((unsigned short)(u >> 16))));
}

__device__ __forceinline__ unsigned pack2(float x, float y) {
    __half2 h = __floats2half2_rn(x, y);
    return *(unsigned*)&h;
}

__device__ __forceinline__ float2 lds2(const __half* p) {
    return word2(*(const unsigned*)p);
}

union HalfWord {
    unsigned u;
    __half2 h;
};

__device__ __forceinline__ unsigned gelu_word(float x, float y) {
    HalfWord v;
    v.h = __floats2half2_rn(x, y);
    const __half2 k = __float2half2_rn(0.7978845608f), c = __float2half2_rn(0.044715f);
    const __half2 half = __float2half2_rn(0.5f), one = __float2half2_rn(1.0f);
    __half2 x2 = __hmul2(v.h, v.h);
    __half2 inner = __hmul2(k, __hfma2(__hmul2(c, x2), v.h, v.h));
    HalfWord t;
    asm("tanh.approx.f16x2 %0, %1;" : "=r"(t.u) : "r"(*(unsigned*)&inner));
    HalfWord out;
    out.h = __hmul2(__hmul2(half, v.h), __hadd2(one, t.h));
    return out.u;
}

__device__ __forceinline__ void frag_a16(const __half* a, int row, int k, int lane,
                                         unsigned (&f)[4]) {
    const __half* p = a + (size_t)(row + (lane & 7) + 8 * ((lane >> 3) & 1)) * TLDS + k + 8 * (lane >> 4);
    unsigned addr = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3]) : "r"(addr));
}

__device__ __forceinline__ void frag_b16(const unsigned* __restrict__ w, int kt, int nt, int ntiles,
                                         int lane, unsigned (&f)[2]) {
    const unsigned* p = w + (((size_t)kt * ntiles + nt) * 32 + lane) * 2;
    const uint2 v = *(const uint2*)p;
    f[0] = v.x;
    f[1] = v.y;
}

template <int PARTS>
__device__ __forceinline__ void trunk_gemm(const __half* a, const unsigned* __restrict__ w,
                                           int warp, int lane,
                                           unsigned (&acc)[PARTS][TMT][TWN][2]) {
#pragma unroll
    for (int p = 0; p < PARTS; ++p)
#pragma unroll
        for (int m = 0; m < TMT; ++m)
#pragma unroll
            for (int n = 0; n < TWN; ++n) acc[p][m][n][0] = acc[p][m][n][1] = 0u;
    unsigned bf[PARTS][TWN][2], bn[PARTS][TWN][2];
#pragma unroll
    for (int p = 0; p < PARTS; ++p)
#pragma unroll
        for (int n = 0; n < TWN; ++n)
            frag_b16(w, 0, p * TNT + warp * TWN + n, PARTS * TNT, lane, bf[p][n]);
    for (int kt = 0; kt < TRUNK_C / 16; ++kt) {
        if (kt + 1 < TRUNK_C / 16)
#pragma unroll
            for (int p = 0; p < PARTS; ++p)
#pragma unroll
                for (int n = 0; n < TWN; ++n)
                    frag_b16(w, kt + 1, p * TNT + warp * TWN + n, PARTS * TNT, lane, bn[p][n]);
#pragma unroll
        for (int m = 0; m < TMT; ++m) {
            unsigned af[4];
            frag_a16(a, 16 * m, 16 * kt, lane, af);
#pragma unroll
            for (int p = 0; p < PARTS; ++p)
#pragma unroll
                for (int n = 0; n < TWN; ++n) mma_h16(acc[p][m][n], af, bf[p][n]);
        }
#pragma unroll
        for (int p = 0; p < PARTS; ++p)
#pragma unroll
            for (int n = 0; n < TWN; ++n) {
                bf[p][n][0] = bn[p][n][0];
                bf[p][n][1] = bn[p][n][1];
            }
    }
}

__device__ __forceinline__ void store_tiles(__half* dst, const unsigned (&acc)[TMT][TWN][2],
                                            int warp, int lane) {
#pragma unroll
    for (int m = 0; m < TMT; ++m)
#pragma unroll
        for (int n = 0; n < TWN; ++n) {
            int col = 8 * (warp * TWN + n) + 2 * (lane & 3);
            *(unsigned*)(dst + (size_t)(16 * m + (lane >> 2)) * TLDS + col) = acc[m][n][0];
            *(unsigned*)(dst + (size_t)(16 * m + (lane >> 2) + 8) * TLDS + col) = acc[m][n][1];
        }
}

struct TrunkShared {
    __half a[TPROWS * TLDS];
    __half u[TPROWS * TLDS];
    __half pooled[TB * GLDS];
    float gb[TB][TRUNK_C];
    int near[N_HEXES * 6];
};

__device__ __forceinline__ void load_row(const __half* p, float (&v)[TQ][2], int lane) {
#pragma unroll
    for (int q = 0; q < TQ; ++q) {
        float2 t = lds2(p + 2 * (lane + 32 * q));
        v[q][0] = t.x;
        v[q][1] = t.y;
    }
}

__device__ __forceinline__ void norm_rows(float (&v)[2][TQ][2], unsigned (&o)[2][TQ],
                                          const float* __restrict__ g,
                                          const float* __restrict__ b, int lane) {
    float sum[2] = {0.0f, 0.0f}, sq[2] = {0.0f, 0.0f};
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int q = 0; q < TQ; ++q) {
            sum[r] += v[r][q][0] + v[r][q][1];
            sq[r] += v[r][q][0] * v[r][q][0] + v[r][q][1] * v[r][q][1];
        }
    for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            sum[r] += __shfl_xor_sync(0xffffffff, sum[r], o);
            sq[r] += __shfl_xor_sync(0xffffffff, sq[r], o);
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        float mean = sum[r] / TRUNK_C;
        float inv = rsqrtf(fmaxf(sq[r] / TRUNK_C - mean * mean, 0.0f) + 1e-5f);
#pragma unroll
        for (int q = 0; q < TQ; ++q) {
            int col = 2 * (lane + 32 * q);
            o[r][q] = gelu_word((v[r][q][0] - mean) * inv * g[col] + b[col],
                                (v[r][q][1] - mean) * inv * g[col + 1] + b[col + 1]);
        }
    }
}

__device__ __forceinline__ void store_words(__half* p, const unsigned (&o)[TQ], int lane) {
#pragma unroll
    for (int q = 0; q < TQ; ++q) *(unsigned*)(p + 2 * (lane + 32 * q)) = o[q];
}

__device__ __forceinline__ void pool_pass(TrunkShared& s, const float* __restrict__ g,
                                          const float* __restrict__ b, int warp, int lane,
                                          int tid, float* pool) {
    const int board = warp % TB, slice = warp / TB, slices = TWARPS / TB;
    float ps[TQ][2], pm[TQ][2];
#pragma unroll
    for (int q = 0; q < TQ; ++q) {
        ps[q][0] = ps[q][1] = 0.0f;
        pm[q][0] = pm[q][1] = neg_inf();
    }
    for (int hex = slice; hex < N_HEXES; hex += 2 * slices) {
        __half* p0 = s.a + (size_t)(board * N_HEXES + hex) * TLDS;
        __half* p1 = p0 + (size_t)slices * TLDS;
        const bool two = hex + slices < N_HEXES;
        float v[2][TQ][2];
        unsigned o[2][TQ];
        load_row(p0, v[0], lane);
        load_row(two ? p1 : p0, v[1], lane);
        norm_rows(v, o, g, b, lane);
        store_words(p0, o[0], lane);
        if (two) store_words(p1, o[1], lane);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (r && !two) break;
#pragma unroll
            for (int q = 0; q < TQ; ++q) {
                float2 y = word2(o[r][q]);
                ps[q][0] += y.x;
                ps[q][1] += y.y;
                pm[q][0] = fmaxf(pm[q][0], y.x);
                pm[q][1] = fmaxf(pm[q][1], y.y);
            }
        }
    }
    float* part = (float*)s.u + (size_t)warp * 2 * TRUNK_C;
#pragma unroll
    for (int q = 0; q < TQ; ++q) {
        int col = 2 * (lane + 32 * q);
        *(float2*)(part + col) = make_float2(ps[q][0], ps[q][1]);
        *(float2*)(part + TRUNK_C + col) = make_float2(pm[q][0], pm[q][1]);
    }
    __syncthreads();
    for (int i = tid; i < TB * 2 * TRUNK_C; i += 32 * TWARPS) {
        int k = i / (2 * TRUNK_C), j = i - k * 2 * TRUNK_C;
        const float* src = (const float*)s.u + j;
        float acc = j < TRUNK_C ? 0.0f : neg_inf();
        for (int w = k; w < TWARPS; w += TB) {
            float v = src[(size_t)w * 2 * TRUNK_C];
            acc = j < TRUNK_C ? acc + v : fmaxf(acc, v);
        }
        if (j < TRUNK_C) acc /= N_HEXES;
        s.pooled[k * GLDS + j] = __float2half(acc);
        if (pool) pool[k * 2 * TRUNK_C + j] = acc;
    }
}

struct Stem {
    const float* cards;
    const int* card_of_row;
    const float* w_pile;
    const float* seat;
    const float* w_tok;
    const unsigned* w_hex;
    const float* b_hex;
    const float* pos;
    const float* w_glob;
};

static_assert(TB * NTYPE * (TOKEN_W + TRUNK_C) + 2 * TB * TRUNK_C <= TPROWS * TLDS / 2,
              "the stem scratch fits in the neighbour buffer");
static_assert(HEX_K <= TRUNK_C, "the hex facts tile fits an activation row");

__device__ void stem(TrunkShared& s, const Stem& st, const float* __restrict__ xpub, __half* tokens,
                     int first, int nboards, float (&x)[TMT][TWN][4]) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float* tk = (float*)s.u;
    float* pj = tk + TB * NTYPE * TOKEN_W;
    float* tp = pj + TB * NTYPE * TRUNK_C;
    float* gl = tp + TB * TRUNK_C;
    for (int i = tid; i < nboards * NTYPE * TOKEN_W; i += 32 * TWARPS) {
        int b = i / (NTYPE * TOKEN_W), t = (i / TOKEN_W) % NTYPE, j = i % TOKEN_W;
        const float* xp = xpub + (size_t)(first + b) * PUBFEAT + OFF_PILES + t * PILE_COUNTS;
        float acc = st.cards[((size_t)st.card_of_row[first + b] * NTYPE + t) * TOKEN_W + j]
                  + st.seat[(size_t)(t / NSLOT) * TOKEN_W + j];
        for (int k = 0; k < PILE_COUNTS; ++k) acc += xp[k] * st.w_pile[(size_t)k * TOKEN_W + j];
        tk[i] = acc;
    }
    for (int row = warp; row < TPROWS; row += TWARPS) {
        int b = row / N_HEXES, hex = row - b * N_HEXES;
        const float* xh = xpub + (size_t)(first + b) * PUBFEAT + hex * HEX_CH;
        for (int k = lane; k < HEX_K; k += 32)
            s.a[(size_t)row * TLDS + k] = __float2half(row < nboards * N_HEXES && k < HEX_FACTS ? xh[k] : 0.0f);
    }
    __syncthreads();
    const int half = tid / TRUNK_C, c = tid % TRUNK_C;
    for (int bt = half; bt < nboards * NTYPE; bt += 32 * TWARPS / TRUNK_C) {
        const float* row = tk + (size_t)bt * TOKEN_W;
        float acc = 0.0f;
#pragma unroll 8
        for (int j = 0; j < TOKEN_W; ++j) acc += row[j] * st.w_tok[(size_t)j * TRUNK_C + c];
        pj[bt * TRUNK_C + c] = acc;
        tokens[((size_t)first * NTYPE + bt) * TRUNK_C + c] = __float2half(acc);
    }
    __syncthreads();
    for (int i = tid; i < nboards * TRUNK_C; i += 32 * TWARPS) {
        int b = i / TRUNK_C, c = i % TRUNK_C;
        float acc = 0.0f;
        for (int t = 0; t < NTYPE; ++t) acc += gelu1(pj[((size_t)b * NTYPE + t) * TRUNK_C + c]);
        tp[i] = acc / NTYPE;
        const float* loose = xpub + (size_t)(first + b) * PUBFEAT + OFF_LOOSE;
        float g = 0.0f;
        for (int k = 0; k < LOOSE; ++k) g += loose[k] * st.w_glob[(size_t)k * TRUNK_C + c];
        gl[i] = g;
    }
    __syncthreads();
    unsigned acc[TMT][TWN][2];
#pragma unroll
    for (int m = 0; m < TMT; ++m)
#pragma unroll
        for (int n = 0; n < TWN; ++n) acc[m][n][0] = acc[m][n][1] = 0u;
    for (int kt = 0; kt < HEX_K / 16; ++kt) {
        unsigned bf[TWN][2];
#pragma unroll
        for (int n = 0; n < TWN; ++n) frag_b16(st.w_hex, kt, warp * TWN + n, TNT, lane, bf[n]);
#pragma unroll
        for (int m = 0; m < TMT; ++m) {
            unsigned af[4];
            frag_a16(s.a, 16 * m, 16 * kt, lane, af);
#pragma unroll
            for (int n = 0; n < TWN; ++n) mma_h16(acc[m][n], af, bf[n]);
        }
    }
#pragma unroll
    for (int m = 0; m < TMT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int row = 16 * m + (lane >> 2) + 8 * h;
            int b = row / N_HEXES, hex = row - b * N_HEXES;
            const bool live = row < nboards * N_HEXES;
            const float* xh = xpub + (size_t)(first + b) * PUBFEAT + hex * HEX_CH;
            int occ = -1;
            if (live)
                for (int t = 0; t < NTYPE; ++t)
                    if (xh[HEX_FACTS + t] != 0.0f) { occ = t; break; }
#pragma unroll
            for (int n = 0; n < TWN; ++n) {
                int col = 8 * (warp * TWN + n) + 2 * (lane & 3);
                float2 f = word2(acc[m][n][h]);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    float v = 0.0f;
                    if (live) {
                        int cc = col + j;
                        v = (j ? f.y : f.x) + st.b_hex[cc] + st.pos[(size_t)hex * TRUNK_C + cc]
                          + gl[b * TRUNK_C + cc] + tp[b * TRUNK_C + cc];
                        if (occ >= 0) v += pj[((size_t)b * NTYPE + occ) * TRUNK_C + cc];
                    }
                    x[m][n][2 * h + j] = v;
                }
            }
        }
    __syncthreads();
}

extern "C" __global__ void __launch_bounds__(32 * TWARPS, 1)
k_trunk(const float* __restrict__ xpub, const Stem st, const unsigned* __restrict__ w,
        const uint4* __restrict__ wg, const float* __restrict__ bias,
        const float* __restrict__ ln, const int* __restrict__ near,
        float* out, __half* spatial, __half* tokens, int boards) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    TrunkShared& s = *(TrunkShared*)smem_raw;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int first = blockIdx.x * TB;
    const int nboards = min(boards - first, TB);
    const int rows = nboards * N_HEXES;
    for (int i = tid; i < N_HEXES * 6; i += 32 * TWARPS) s.near[i] = near[i];

    float x[TMT][TWN][4];
    stem(s, st, xpub, tokens, first, nboards, x);

    for (int blk = 0; blk < BLOCKS; ++blk) {
        const unsigned* w1 = w + (size_t)blk * 3 * TPACK;
        const unsigned* w2 = w1 + 2 * TPACK;
        const float* b1 = bias + (size_t)blk * 2 * TRUNK_C;
        const float* b2 = b1 + TRUNK_C;
        const float* l = ln + (size_t)blk * 4 * TRUNK_C;
        unsigned acc[2][TMT][TWN][2];
#pragma unroll
        for (int m = 0; m < TMT; ++m)
#pragma unroll
            for (int n = 0; n < TWN; ++n) {
                acc[0][m][n][0] = pack2(x[m][n][0], x[m][n][1]);
                acc[0][m][n][1] = pack2(x[m][n][2], x[m][n][3]);
            }
        store_tiles(s.a, acc[0], warp, lane);
        __syncthreads();
        pool_pass(s, l, l + TRUNK_C, warp, lane, tid, nullptr);
        __syncthreads();
        if (warp < TRUNK_C / 16) {
            const uint4* wgp = wg + (size_t)blk * (2 * TRUNK_C / 16) * (TRUNK_C / 16) * 32;
            unsigned d[6][2] = {};
            const int g = lane >> 2, t = lane & 3;
            const uint4* wl = wgp + (size_t)warp * (2 * TRUNK_C / 16) * 32 + lane;
            for (int k0 = 0; k0 < 2 * TRUNK_C / 16; k0 += 6) {
                uint4 av[6];
#pragma unroll
                for (int i = 0; i < 6; ++i) av[i] = wl[(size_t)(k0 + i) * 32];
#pragma unroll
                for (int i = 0; i < 6; ++i) {
                    unsigned af[4] = {av[i].x, av[i].y, av[i].z, av[i].w};
                    unsigned bf[2] = {0u, 0u};
                    if (g < TB) {
                        const __half* p = s.pooled + (size_t)g * GLDS + 16 * (k0 + i) + 2 * t;
                        bf[0] = *(const unsigned*)p;
                        bf[1] = *(const unsigned*)(p + 8);
                    }
                    mma_h16(d[i], af, bf);
                }
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int ch = 16 * warp + g + 8 * h;
                float2 v = make_float2(0.0f, 0.0f);
#pragma unroll
                for (int i = 0; i < 6; ++i) {
                    float2 t2 = word2(d[i][h]);
                    v.x += t2.x;
                    v.y += t2.y;
                }
                if (2 * t < TB) s.gb[2 * t][ch] = v.x + b1[ch];
                if (2 * t + 1 < TB) s.gb[2 * t + 1][ch] = v.y + b1[ch];
            }
        }
        trunk_gemm<2>(s.a, w1, warp, lane, acc);
        __syncthreads();
        store_tiles(s.u, acc[0], warp, lane);
        store_tiles(s.a, acc[1], warp, lane);
        __syncwarp();
        if (lane < 8) *(unsigned*)(s.u + (size_t)(TPROWS - 1) * TLDS + 16 * warp + 2 * lane) = 0u;
        __syncthreads();
        for (int row0 = warp; row0 < TROWS; row0 += 2 * TWARPS) {
            const bool two = row0 + TWARPS < TROWS;
            float v[2][TQ][2];
            unsigned o[2][TQ];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                int row = two || !r ? row0 + r * TWARPS : row0;
                int board = row / N_HEXES, hex = row - board * N_HEXES;
                HalfWord nb[TQ];
#pragma unroll
                for (int q = 0; q < TQ; ++q) nb[q].u = 0u;
#pragma unroll
                for (int k = 0; k < 6; ++k) {
                    int n = s.near[hex * 6 + k];
                    int urow = n < 0 ? TPROWS - 1 : board * N_HEXES + n;
                    const __half* u = s.u + (size_t)urow * TLDS;
#pragma unroll
                    for (int q = 0; q < TQ; ++q) {
                        HalfWord t;
                        t.u = *(const unsigned*)(u + 2 * (lane + 32 * q));
                        nb[q].h = __hadd2(nb[q].h, t.h);
                    }
                }
                load_row(s.a + (size_t)row * TLDS, v[r], lane);
#pragma unroll
                for (int q = 0; q < TQ; ++q) {
                    int col = 2 * (lane + 32 * q);
                    float2 t = __half22float2(nb[q].h);
                    v[r][q][0] += s.gb[board][col] + t.x;
                    v[r][q][1] += s.gb[board][col + 1] + t.y;
                }
            }
            norm_rows(v, o, l + 2 * TRUNK_C, l + 3 * TRUNK_C, lane);
            store_words(s.a + (size_t)row0 * TLDS, o[0], lane);
            if (two) store_words(s.a + (size_t)(row0 + TWARPS) * TLDS, o[1], lane);
        }
        __syncthreads();
        unsigned (&out2)[1][TMT][TWN][2] = *(unsigned (*)[1][TMT][TWN][2])&acc[0];
        trunk_gemm<1>(s.a, w2, warp, lane, out2);
#pragma unroll
        for (int m = 0; m < TMT; ++m)
#pragma unroll
            for (int n = 0; n < TWN; ++n) {
                int col = 8 * (warp * TWN + n) + 2 * (lane & 3);
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    float2 v = word2(acc[0][m][n][h]);
                    x[m][n][2 * h] += v.x + b2[col];
                    x[m][n][2 * h + 1] += v.y + b2[col + 1];
                }
            }
        __syncthreads();
    }

    unsigned acc[TMT][TWN][2];
#pragma unroll
    for (int m = 0; m < TMT; ++m)
#pragma unroll
        for (int n = 0; n < TWN; ++n) {
            acc[m][n][0] = pack2(x[m][n][0], x[m][n][1]);
            acc[m][n][1] = pack2(x[m][n][2], x[m][n][3]);
        }
    store_tiles(s.a, acc, warp, lane);
    __syncthreads();
    const float* l = ln + (size_t)BLOCKS * 4 * TRUNK_C;
    float* pool = (float*)s.u + TWARPS * 2 * TRUNK_C;
    pool_pass(s, l, l + TRUNK_C, warp, lane, tid, pool);
    __syncthreads();
    const size_t width = 2 * TRUNK_C + LOOSE;
    for (int i = tid; i < rows * TRUNK_C / 2; i += 32 * TWARPS) {
        int row = i / (TRUNK_C / 2), col = 2 * (i - row * (TRUNK_C / 2));
        *(unsigned*)(spatial + ((size_t)first * N_HEXES + row) * TRUNK_C + col) =
            *(const unsigned*)(s.a + (size_t)row * TLDS + col);
    }
    for (int i = tid; i < TB * (int)width; i += 32 * TWARPS) {
        int k = i / width, j = i - k * width;
        if (first + k >= boards) break;
        out[(first + k) * width + j] = j < 2 * TRUNK_C
            ? pool[k * 2 * TRUNK_C + j]
            : xpub[(size_t)(first + k) * PUBFEAT + OFF_LOOSE + j - 2 * TRUNK_C];
    }
}
