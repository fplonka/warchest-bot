static_assert(HIDDEN + TRUNK_C == D, "the mid layer holds the value hidden layer and the action board term");
static_assert(32 % BINS == 0 && HR * BINS == 256 && TRUNK_C % 32 == 0,
              "a thread holds one value bin of one row and a warp one action row");

struct Head {
    const float* w_value;
    const float* b_value;
    const float* kind;
    const float* role;
    const float* gamma;
    const float* beta;
    const float* w_logit;
};

__device__ float action_logit(const Head& h, const unsigned char* d, const __half* act, const __half* tokens,
                              const __half* spatial, int lane) {
    const __half* src[5];
    float on[5];
#pragma unroll
    for (int r = 0; r < 5; ++r) {
        on[r] = d[r + 1] < (r < 2 ? NTYPE : N_HEXES);
        src[r] = r < 2 ? tokens + (size_t)(on[r] ? d[r + 1] : 0) * TRUNK_C
                       : spatial + (size_t)(on[r] ? d[r + 1] : 0) * TRUNK_C;
    }
    const float* kind = h.kind + (size_t)d[0] * TRUNK_C;
    float v[TRUNK_C / 32];
    float sum = 0.0f;
#pragma unroll
    for (int q = 0; q < TRUNK_C / 32; ++q) {
        const int j = lane + 32 * q;
        float x = kind[j] + __half2float(act[j]);
#pragma unroll
        for (int r = 0; r < 5; ++r) x += on[r] * h.role[r * TRUNK_C + j] * __half2float(src[r][j]);
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

extern "C" __global__ void __launch_bounds__(256)
k_heads(const __half* __restrict__ mid, const Head h, const unsigned char* __restrict__ desc,
        const unsigned* __restrict__ off, const __half* __restrict__ tokens, const __half* __restrict__ spatial,
        float* out, int n) {
    __shared__ float hidden[HR][HIDDEN];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int first = blockIdx.x * HR, rows = min(n - first, HR);
    for (int i = tid; i < HR * HIDDEN; i += 256) {
        const int r = i / HIDDEN, k = i - r * HIDDEN;
        hidden[r][k] = r < rows ? gelu1(__half2float(mid[(size_t)(first + r) * D + k])) : 0.0f;
    }
    __syncthreads();
    {
        const int r = tid / BINS, bin = tid % BINS;
        float logit = h.b_value[bin];
#pragma unroll 8
        for (int k = 0; k < HIDDEN; ++k) logit += hidden[r][k] * h.w_value[k * BINS + bin];
        float top = logit;
        for (int m = BINS / 2; m > 0; m >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffff, top, m, BINS));
        float e = expf(logit - top), centre = e * ((bin + 0.5f) / (BINS / 2) - 1.0f);
        for (int m = BINS / 2; m > 0; m >>= 1) {
            e += __shfl_xor_sync(0xffffffff, e, m, BINS);
            centre += __shfl_xor_sync(0xffffffff, centre, m, BINS);
        }
        if (bin == 0 && r < rows) out[first + r] = centre / e;
    }
    float* prior = out + n;
    for (unsigned i = off[first] + warp; i < off[first + rows]; i += 8) {
        int r = 0;
        while (off[first + r + 1] <= i) ++r;
        const size_t row = first + r;
        const float logit = action_logit(h, desc + (size_t)i * ACT_BYTES, mid + row * D + HIDDEN,
                                         tokens + row * NTYPE * TRUNK_C, spatial + row * N_HEXES * TRUNK_C, lane);
        if (lane == 0) prior[i] = logit;
    }
    __syncthreads();
    for (int r = warp; r < rows; r += 8) {
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
