#define POOLED (2 * TRUNK_C + LOOSE)
#define HK ((POOLED + 15) / 16 * 16)
#define HA (HK + 8)
#define HD (D + 8)
#define HNT (D / 8 / TWARPS)
static_assert(D == 8 * HNT * TWARPS && HIDDEN + TRUNK_C == D, "a warp owns whole column tiles of both head layers");
static_assert(HD <= HA, "the board rows reuse the pooled rows");
static_assert(BINS <= 32 && TRUNK_C % 32 == 0, "a warp holds the value bins and an action row");

struct Head {
    const unsigned* w_board;
    const float* b_board;
    const unsigned* w_mid;
    const float* b_hidden;
    const float* w_value;
    const float* b_value;
    const float* kind;
    const float* role;
    const float* gamma;
    const float* beta;
    const float* w_logit;
};

__device__ __forceinline__ void mma_f32(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int LDS, int KT>
__device__ __forceinline__ void head_gemm(const __half* a, const unsigned* __restrict__ w, int warp, int lane,
                                          float (&acc)[HNT][4]) {
#pragma unroll
    for (int n = 0; n < HNT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
#pragma unroll 4
    for (int kt = 0; kt < KT; ++kt) {
        unsigned af[4];
        frag_a16<LDS>(a, 0, 16 * kt, lane, af);
#pragma unroll
        for (int n = 0; n < HNT; ++n) {
            unsigned bf[2];
            frag_b16(w, kt, warp * HNT + n, D / 8, lane, bf);
            mma_f32(acc[n], af, bf);
        }
    }
}

__device__ float action_logit(const Head& h, const unsigned char* d, const float* act, const __half* tokens,
                              const __half* spatial, int lane) {
    float v[TRUNK_C / 32];
    float sum = 0.0f;
#pragma unroll
    for (int q = 0; q < TRUNK_C / 32; ++q) {
        const int j = lane + 32 * q;
        float x = h.kind[(size_t)d[0] * TRUNK_C + j] + act[j];
#pragma unroll
        for (int r = 0; r < 5; ++r) {
            const int e = d[r + 1];
            const __half* src = r < 2 ? (e < NTYPE ? tokens + (size_t)e * TRUNK_C : nullptr)
                                      : (e < N_HEXES ? spatial + (size_t)e * TRUNK_C : nullptr);
            if (src) x += h.role[r * TRUNK_C + j] * __half2float(src[j]);
        }
        v[q] = x;
        sum += x;
    }
    const float mean = warp_sum(sum) / TRUNK_C;
    float var = 0.0f;
#pragma unroll
    for (int q = 0; q < TRUNK_C / 32; ++q) var += (v[q] - mean) * (v[q] - mean);
    const float inv = rsqrtf(warp_sum(var) / TRUNK_C + 1e-5f);
    float acc = 0.0f;
#pragma unroll
    for (int q = 0; q < TRUNK_C / 32; ++q) {
        const int j = lane + 32 * q;
        acc += gelu1((v[q] - mean) * inv * h.gamma[j] + h.beta[j]) * h.w_logit[j];
    }
    return warp_sum(acc);
}

extern "C" __global__ void __launch_bounds__(32 * TWARPS)
k_heads(const float* __restrict__ pooled, const Head h, const unsigned char* __restrict__ desc,
        const unsigned* __restrict__ off, const __half* __restrict__ tokens, const __half* __restrict__ spatial,
        float* out, int n) {
    __shared__ __align__(16) __half a[HR * HA];
    __shared__ float mid[HR][D];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int first = blockIdx.x * HR, rows = min(n - first, HR);
    for (int i = tid; i < HR * HK; i += 32 * TWARPS) {
        const int r = i / HK, k = i - r * HK;
        a[r * HA + k] = __float2half(r < rows && k < POOLED ? pooled[(size_t)(first + r) * POOLED + k] : 0.0f);
    }
    __syncthreads();
    float acc[HNT][4];
    head_gemm<HA, HK / 16>(a, h.w_board, warp, lane, acc);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < HNT; ++j) {
        const int col = 8 * (warp * HNT + j) + 2 * t;
        const float b0 = h.b_board[col], b1 = h.b_board[col + 1];
        *(unsigned*)(a + g * HD + col) = pack2(acc[j][0] + b0, acc[j][1] + b1);
        *(unsigned*)(a + (g + 8) * HD + col) = pack2(acc[j][2] + b0, acc[j][3] + b1);
    }
    __syncthreads();
    head_gemm<HD, D / 16>(a, h.w_mid, warp, lane, acc);
#pragma unroll
    for (int j = 0; j < HNT; ++j) {
        const int col = 8 * (warp * HNT + j) + 2 * t;
        const float b0 = col < HIDDEN ? h.b_hidden[col] : 0.0f, b1 = col < HIDDEN ? h.b_hidden[col + 1] : 0.0f;
        mid[g][col] = acc[j][0] + b0;
        mid[g][col + 1] = acc[j][1] + b1;
        mid[g + 8][col] = acc[j][2] + b0;
        mid[g + 8][col + 1] = acc[j][3] + b1;
    }
    __syncthreads();
    for (int r = warp; r < rows; r += TWARPS) {
        float logit = neg_inf();
        if (lane < BINS) {
            logit = h.b_value[lane];
            for (int k = 0; k < HIDDEN; ++k) logit += gelu1(mid[r][k]) * h.w_value[k * BINS + lane];
        }
        float top = logit;
        for (int m = 16; m > 0; m >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffff, top, m));
        const float e = lane < BINS ? expf(logit - top) : 0.0f;
        const float centre = (lane + 0.5f) / (BINS / 2) - 1.0f;
        const float total = warp_sum(e), mean = warp_sum(e * centre);
        if (lane == 0) out[first + r] = mean / total;
    }
    float* prior = out + n;
    for (unsigned i = off[first] + warp; i < off[first + rows]; i += TWARPS) {
        int r = 0;
        while (off[first + r + 1] <= i) ++r;
        const size_t row = first + r;
        const float logit = action_logit(h, desc + (size_t)i * ACT_BYTES, mid[r] + HIDDEN,
                                         tokens + row * NTYPE * TRUNK_C, spatial + row * N_HEXES * TRUNK_C, lane);
        if (lane == 0) prior[i] = logit;
    }
    __syncthreads();
    for (int r = warp; r < rows; r += TWARPS) {
        const unsigned lo = off[first + r], hi = off[first + r + 1];
        float top = neg_inf();
        for (unsigned i = lo + lane; i < hi; i += 32) top = fmaxf(top, prior[i]);
        for (int m = 16; m > 0; m >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffff, top, m));
        float mine = 0.0f;
        for (unsigned i = lo + lane; i < hi; i += 32) {
            const float e = expf(prior[i] - top);
            prior[i] = e;
            mine += e;
        }
        const float scale = 1.0f / warp_sum(mine);
        for (unsigned i = lo + lane; i < hi; i += 32) prior[i] *= scale;
    }
}
