
#define TB 3
#define TROWS (TB * N_HEXES)
#define TMT ((TROWS + 15) / 16)
#define TPROWS (TMT * 16)
#define TPITCH ((N_HEXES + 15) / 16 * 16)
#define TKT (TB * TPITCH / 16)
#define TSTEM (TB * NTYPE * (TOKEN_W + TRUNK_C) + 3 * TB * TRUNK_C + TB * LOOSE)
#define TBUF (TB * TPITCH * TLDS > TPROWS * TLDS + 2 * TSTEM ? TB * TPITCH * TLDS : TPROWS * TLDS + 2 * TSTEM)
#define TNT (TRUNK_C / 8)
#define TWN (TNT / TWARPS)
#define TLDS (TRUNK_C + 8)
#define TPACK (TRUNK_C * TRUNK_C / 2)
#define GLDS (2 * TRUNK_C + 8)
static_assert(TWN == 2, "the neighbour sum reads a warp's two column tiles with one transposed ldmatrix");
static_assert(TRUNK_C / 16 == TWARPS, "the pooled projection gives every warp one channel tile");
static_assert(HEX_CH <= 32 && HEX_K == 32, "the hex facts pass reads a hex with one lane a channel");

struct TrunkShared {
    __half a[TBUF];
    __half pooled[TB * GLDS];
    float gb[TB][TRUNK_C];
    float2 stats[TPROWS];
    int occ[TPROWS];
};

__device__ __forceinline__ void mma_h16(unsigned (&d)[2], const unsigned (&a)[4],
                                        const unsigned (&b)[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
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

__device__ __forceinline__ void pool_gemm(TrunkShared& s, const uint4* __restrict__ wg,
                                          const float* __restrict__ b1, int warp, int lane) {
    unsigned d[2][2] = {};
    const int g = lane >> 2, t = lane & 3;
    const uint4* wl = wg + (size_t)warp * (2 * TRUNK_C / 16) * 32 + lane;
    for (int k0 = 0; k0 < 2 * TRUNK_C / 16; k0 += 12) {
        uint4 av[12];
#pragma unroll
        for (int i = 0; i < 12; ++i) av[i] = wl[(size_t)(k0 + i) * 32];
#pragma unroll
        for (int i = 0; i < 12; ++i) {
            unsigned af[4] = {av[i].x, av[i].y, av[i].z, av[i].w};
            unsigned bf[2] = {0u, 0u};
            if (g < TB) {
                const __half* p = s.pooled + (size_t)g * GLDS + 16 * (k0 + i) + 2 * t;
                bf[0] = *(const unsigned*)p;
                bf[1] = *(const unsigned*)(p + 8);
            }
            mma_h16(d[i & 1], af, bf);
        }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int ch = 16 * warp + g + 8 * h;
        const float2 v0 = word2(d[0][h]), v1 = word2(d[1][h]);
        if (2 * t < TB) s.gb[2 * t][ch] = v0.x + v1.x + b1[ch];
        if (2 * t + 1 < TB) s.gb[2 * t + 1][ch] = v0.y + v1.y + b1[ch];
    }
}

template <int PITCH>
__device__ __forceinline__ void store_tiles(__half* dst, const unsigned (&acc)[TMT][TWN][2],
                                            int warp, int lane) {
#pragma unroll
    for (int m = 0; m < TMT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = 16 * m + (lane >> 2) + 8 * h;
            if (PITCH != N_HEXES && r >= TROWS) continue;
            __half* row = dst + (size_t)(r + (PITCH - N_HEXES) * (r / N_HEXES)) * TLDS;
#pragma unroll
            for (int n = 0; n < TWN; ++n)
                *(unsigned*)(row + 8 * (warp * TWN + n) + 2 * (lane & 3)) = acc[m][n][h];
        }
}

__device__ __forceinline__ void frag_b16t(const __half* u, int kt, int n0, int lane, unsigned (&f)[4]) {
    const __half* p = u + (size_t)(16 * kt + (lane & 7) + 8 * ((lane >> 3) & 1)) * TLDS + n0 + 8 * (lane >> 4);
    unsigned addr = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3]) : "r"(addr));
}

__device__ __forceinline__ constexpr bool tiles_touch(int mt, int kt) {
    return (16 * mt) / N_HEXES <= kt / (TPITCH / 16) && kt / (TPITCH / 16) <= (16 * mt + 15) / N_HEXES;
}

__device__ __forceinline__ void zero_pitch(__half* dst, int tid) {
    constexpr int WORDS = (TPITCH - N_HEXES) * TLDS / 2;
    for (int i = tid; i < TB * WORDS; i += 32 * TWARPS) {
        const int b = i / WORDS, w = i - b * WORDS;
        ((unsigned*)(dst + (size_t)(b * TPITCH + N_HEXES) * TLDS))[w] = 0u;
    }
}

static_assert(TWARPS * TPROWS * sizeof(float2) <= TPROWS * TLDS * sizeof(__half),
              "the row statistics partials fit in the activation rows");
static_assert(TPROWS <= TB * TPITCH, "the padded neighbour rows hold the activation rows");

__device__ __forceinline__ void frag_stats(const float (&v)[TMT][TWN][4], float2* part, int warp,
                                           int lane) {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int m = 0; m < TMT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float s = 0.0f, q = 0.0f;
#pragma unroll
            for (int n = 0; n < TWN; ++n)
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    float x = v[m][n][2 * h + j];
                    s += x;
                    q += x * x;
                }
            s += __shfl_xor_sync(0xffffffff, s, 1);
            s += __shfl_xor_sync(0xffffffff, s, 2);
            q += __shfl_xor_sync(0xffffffff, q, 1);
            q += __shfl_xor_sync(0xffffffff, q, 2);
            if (t == 0) part[warp * TPROWS + 16 * m + g + 8 * h] = make_float2(s, q);
        }
}

__device__ __forceinline__ void finish_stats(TrunkShared& s, const float2* part, int tid) {
    if (tid >= TPROWS) return;
    float sum = 0.0f, sq = 0.0f;
#pragma unroll
    for (int w = 0; w < TWARPS; ++w) {
        float2 p = part[w * TPROWS + tid];
        sum += p.x;
        sq += p.y;
    }
    float mean = sum / TRUNK_C;
    s.stats[tid] = make_float2(mean, rsqrtf(fmaxf(sq / TRUNK_C - mean * mean, 0.0f) + 1e-5f));
}

__device__ __forceinline__ void norm_frag(const float (&v)[TMT][TWN][4], const TrunkShared& s,
                                          const float* __restrict__ gamma, const float* __restrict__ beta,
                                          int warp, int lane, unsigned (&o)[TMT][TWN][2]) {
    const int g = lane >> 2;
    float gm[TWN][2], bt[TWN][2];
#pragma unroll
    for (int n = 0; n < TWN; ++n)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int col = 16 * warp + 8 * n + 2 * (lane & 3) + j;
            gm[n][j] = gamma[col];
            bt[n][j] = beta[col];
        }
#pragma unroll
    for (int m = 0; m < TMT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float2 st = s.stats[16 * m + g + 8 * h];
#pragma unroll
            for (int n = 0; n < TWN; ++n)
                o[m][n][h] = gelu_word((v[m][n][2 * h] - st.x) * st.y * gm[n][0] + bt[n][0],
                                       (v[m][n][2 * h + 1] - st.x) * st.y * gm[n][1] + bt[n][1]);
        }
}

__device__ __forceinline__ void pool_rows(TrunkShared& s, float* out, int first, int boards, int tid) {
    for (int i = tid; i < TB * TRUNK_C; i += 32 * TWARPS) {
        const int b = i / TRUNK_C, c = i - b * TRUNK_C;
        const __half* p = s.a + (size_t)b * N_HEXES * TLDS + c;
        float sum = 0.0f, top = neg_inf();
#pragma unroll
        for (int j = 0; j < N_HEXES; ++j) {
            const float v = __half2float(p[(size_t)j * TLDS]);
            sum += v;
            top = fmaxf(top, v);
        }
        const float mean = sum / N_HEXES;
        s.pooled[b * GLDS + c] = __float2half(mean);
        s.pooled[b * GLDS + TRUNK_C + c] = __float2half(top);
        if (out && first + b < boards) {
            float* row = out + (size_t)(first + b) * (2 * TRUNK_C + LOOSE);
            row[c] = mean;
            row[TRUNK_C + c] = top;
        }
    }
}

__device__ __forceinline__ void gather(const TrunkShared& s, const uint4* __restrict__ adj,
                                       unsigned (&y)[TMT][TWN][2], int warp, int lane,
                                       float (&v)[TMT][TWN][4]) {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int kt = 0; kt < TKT; ++kt) {
        unsigned bf[4];
        frag_b16t(s.a, kt, 16 * warp, lane, bf);
#pragma unroll
        for (int m = 0; m < TMT; ++m) {
            if (!tiles_touch(m, kt)) continue;
            const uint4 a = adj[(m * TKT + kt) * 32 + lane];
            const unsigned af[4] = {a.x, a.y, a.z, a.w};
            const unsigned b0[2] = {bf[0], bf[1]}, b1[2] = {bf[2], bf[3]};
            mma_h16(y[m][0], af, b0);
            mma_h16(y[m][1], af, b1);
        }
    }
#pragma unroll
    for (int m = 0; m < TMT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int board = min((16 * m + g + 8 * h) / N_HEXES, TB - 1);
#pragma unroll
            for (int n = 0; n < TWN; ++n) {
                const int col = 16 * warp + 8 * n + 2 * t;
                const float2 w = word2(y[m][n][h]);
                v[m][n][2 * h] = w.x + s.gb[board][col];
                v[m][n][2 * h + 1] = w.y + s.gb[board][col + 1];
            }
        }
}

__device__ __forceinline__ void layer_norm(TrunkShared& s, const float (&v)[TMT][TWN][4],
                                           const float* __restrict__ g, const float* __restrict__ b,
                                           int warp, int lane, int tid, unsigned (&o)[TMT][TWN][2]) {
    float2* part = (float2*)s.a;
    frag_stats(v, part, warp, lane);
    __syncthreads();
    finish_stats(s, part, tid);
    __syncthreads();
    norm_frag(v, s, g, b, warp, lane, o);
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

static_assert(HEX_K <= TRUNK_C, "the hex facts tile fits an activation row");

__device__ void stem(TrunkShared& s, const Stem& st, const float* __restrict__ xpub, __half* tokens,
                     int first, int nboards, float (&x)[TMT][TWN][4]) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float* tk = (float*)(s.a + (size_t)TPROWS * TLDS);
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
        const float f = row < nboards * N_HEXES && lane < HEX_CH ? xh[lane] : 0.0f;
        s.a[(size_t)row * TLDS + lane] = __float2half(lane < HEX_FACTS ? f : 0.0f);
        const unsigned typed = __ballot_sync(0xffffffff, lane >= HEX_FACTS && f != 0.0f);
        if (lane == 0) s.occ[row] = typed ? __ffs(typed) - 1 - HEX_FACTS : -1;
    }
    __syncthreads();
    constexpr int SPLIT = 32 * TWARPS / TRUNK_C, PER = (TB * NTYPE + SPLIT - 1) / SPLIT;
    static_assert(SPLIT == 2, "the stem splits the loose projection between two halves of the block");
    const int half = tid / TRUNK_C, c = tid % TRUNK_C;
    {
        float acc[PER];
#pragma unroll
        for (int k = 0; k < PER; ++k) acc[k] = 0.0f;
        for (int j = 0; j < TOKEN_W; ++j) {
            const float wv = st.w_tok[(size_t)j * TRUNK_C + c];
#pragma unroll
            for (int k = 0; k < PER; ++k) acc[k] += tk[(size_t)(SPLIT * k + half) * TOKEN_W + j] * wv;
        }
#pragma unroll
        for (int k = 0; k < PER; ++k) {
            const int bt = SPLIT * k + half;
            if (bt < nboards * NTYPE) {
                pj[bt * TRUNK_C + c] = acc[k];
                tokens[((size_t)first * NTYPE + bt) * TRUNK_C + c] = __float2half(acc[k]);
            }
        }
    }
    float* loose = gl + 2 * TB * TRUNK_C;
    for (int i = tid; i < TB * LOOSE; i += 32 * TWARPS) {
        const int b = i / LOOSE, k = i - b * LOOSE;
        loose[i] = b < nboards ? xpub[(size_t)(first + b) * PUBFEAT + OFF_LOOSE + k] : 0.0f;
    }
    __syncthreads();
    {
        float g[TB];
#pragma unroll
        for (int b = 0; b < TB; ++b) g[b] = 0.0f;
        const int k0 = half * (LOOSE / SPLIT), k1 = half == SPLIT - 1 ? LOOSE : k0 + LOOSE / SPLIT;
#pragma unroll 8
        for (int k = k0; k < k1; ++k) {
            const float wv = st.w_glob[(size_t)k * TRUNK_C + c];
#pragma unroll
            for (int b = 0; b < TB; ++b) g[b] += loose[b * LOOSE + k] * wv;
        }
#pragma unroll
        for (int b = 0; b < TB; ++b) gl[(half * TB + b) * TRUNK_C + c] = g[b];
    }
    __syncthreads();
    if (half == 0) {
#pragma unroll
        for (int b = 0; b < TB; ++b) {
            float acc = 0.0f;
#pragma unroll
            for (int t = 0; t < NTYPE; ++t) acc += b < nboards ? gelu1(pj[((size_t)b * NTYPE + t) * TRUNK_C + c]) : 0.0f;
            tp[b * TRUNK_C + c] = acc / NTYPE;
            gl[b * TRUNK_C + c] += gl[(TB + b) * TRUNK_C + c];
        }
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
            const int occ = s.occ[row];
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
        const float* __restrict__ ln, const uint4* __restrict__ adj,
        float* out, __half* spatial, __half* tokens, int boards) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    TrunkShared& s = *(TrunkShared*)smem_raw;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int first = blockIdx.x * TB;
    const int nboards = min(boards - first, TB);
    const int rows = nboards * N_HEXES;

    float x[TMT][TWN][4];
    stem(s, st, xpub, tokens, first, nboards, x);
    zero_pitch(s.a, tid);

    for (int blk = 0; blk < BLOCKS; ++blk) {
        const unsigned* w1 = w + (size_t)blk * 3 * TPACK;
        const unsigned* w2 = w1 + 2 * TPACK;
        const float* b1 = bias + (size_t)blk * 2 * TRUNK_C;
        const float* b2 = b1 + TRUNK_C;
        const float* l = ln + (size_t)blk * 4 * TRUNK_C;
        unsigned o[TMT][TWN][2];
        __syncthreads();
        layer_norm(s, x, l, l + TRUNK_C, warp, lane, tid, o);
        store_tiles<N_HEXES>(s.a, o, warp, lane);
        __syncthreads();
        pool_rows(s, nullptr, first, boards, tid);
        __syncthreads();
        pool_gemm(s, wg + (size_t)blk * (2 * TRUNK_C / 16) * (TRUNK_C / 16) * 32, b1, warp, lane);
        unsigned acc[2][TMT][TWN][2];
        trunk_gemm<2>(s.a, w1, warp, lane, acc);
        __syncthreads();
        store_tiles<TPITCH>(s.a, acc[0], warp, lane);
        __syncthreads();
        float y[TMT][TWN][4];
        gather(s, adj, acc[1], warp, lane, y);
        __syncthreads();
        layer_norm(s, y, l + 2 * TRUNK_C, l + 3 * TRUNK_C, warp, lane, tid, o);
        store_tiles<N_HEXES>(s.a, o, warp, lane);
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
    }

    unsigned o[TMT][TWN][2];
    const float* l = ln + (size_t)BLOCKS * 4 * TRUNK_C;
    __syncthreads();
    layer_norm(s, x, l, l + TRUNK_C, warp, lane, tid, o);
    store_tiles<N_HEXES>(s.a, o, warp, lane);
    __syncthreads();
    pool_rows(s, out, first, boards, tid);
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int m = 0; m < TMT; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = 16 * m + g + 8 * h;
            if (r >= rows) continue;
#pragma unroll
            for (int n = 0; n < TWN; ++n)
                *(unsigned*)(spatial + ((size_t)first * N_HEXES + r) * TRUNK_C + 16 * warp + 8 * n + 2 * t) =
                    o[m][n][h];
        }
    const size_t width = 2 * TRUNK_C + LOOSE;
    for (int i = tid; i < TB * (int)LOOSE; i += 32 * TWARPS) {
        int k = i / LOOSE, j = i - k * LOOSE;
        if (first + k >= boards) break;
        out[(first + k) * width + 2 * TRUNK_C + j] = xpub[(size_t)(first + k) * PUBFEAT + OFF_LOOSE + j];
    }
}
