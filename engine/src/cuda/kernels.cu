#include <cuda_fp16.h>


__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffff, v, s);
    return v;
}


__device__ __forceinline__ float tanh_approx(float x) {
    float r;
    asm("tanh.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float gelu1(float x) {
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanh_approx(inner));
}

__device__ __forceinline__ float2 word2(unsigned u) {
    return make_float2(__half2float(__ushort_as_half((unsigned short)(u & 0xffffu))),
                       __half2float(__ushort_as_half((unsigned short)(u >> 16))));
}

__device__ __forceinline__ unsigned pack2(float x, float y) {
    __half2 h = __floats2half2_rn(x, y);
    return *(unsigned*)&h;
}

union HalfWord {
    unsigned u;
    __half2 h;
};

__device__ __forceinline__ unsigned gelu_h2(unsigned u) {
    HalfWord v;
    v.u = u;
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

__device__ __forceinline__ unsigned gelu_word(float x, float y) {
    return gelu_h2(pack2(x, y));
}

extern "C" __global__ void k_expand_rows(const unsigned char* rows, const float* cards,
                              const unsigned char* locations, float* out, int n) {
    int r = blockIdx.x;
    if (r >= n) return;
    const unsigned char* row = rows + (size_t)r * ROW_BYTES;
    float* dst = out + (size_t)r * PUBFEAT;
    for (int j = threadIdx.x; j < PUBFEAT; j += blockDim.x) {
        float v = 0.0f;
        if (j < HEX_BLOCK) {
            int h = j / HEX_CH, ch = j % HEX_CH;
            int owner = row[ROW_HEX_OWNER + h];
            int slot = row[ROW_HEX_SLOT + h];
            int marker = row[ROW_HEX_MARKER + h];
            if (ch == 0) v = owner == 0;
            else if (ch == 1) v = owner == 1;
            else if (ch == 2) v = owner < 2 ? row[ROW_HEX_HEIGHT + h] / 5.0f : 0.0f;
            else if (ch == 3) v = marker == 0;
            else if (ch == 4) v = marker == 1;
            else if (ch == 5) v = locations[h];
            else if (ch < HEX_FACTS) {
                int d = ch - 6;
                int byte = row[ROW_STACK_OWED + 8 * d + h / 8];
                v = (byte >> (h % 8)) & 1;
            } else {
                int type = ch - HEX_FACTS;
                v = owner < 2 && slot < NSLOT && type == owner * NSLOT + slot;
            }
        } else if (j < OFF_CARDS) {
            v = row[ROW_PILES + j - OFF_PILES] / 5.0f;
        } else if (j < OFF_LOOSE) {
            int k = j - OFF_CARDS;
            int type = k / CARD_FEATS, fact = k % CARD_FEATS;
            v = cards[(size_t)row[ROW_IDS + type] * CARD_FEATS + fact];
        } else if (j < OFF_LOOSE + 2 * PLAYER_SCALARS) {
            int k = j - OFF_LOOSE, player = k / PLAYER_SCALARS;
            int field = k % PLAYER_SCALARS, on_board = 0;
            for (int h = 0; h < N_HEXES; ++h)
                on_board += row[ROW_HEX_MARKER + h] == player;
            if (field == 0) v = (6 - on_board) / 6.0f;
            else if (field == 1) v = on_board / 6.0f;
            else if (field == 2) v = row[ROW_HAND_SIZE + player] / 3.0f;
            else if (field == 3) v = row[ROW_FD_SIZE + player] / MAX_COINS;
            else if (field == 4) v = row[ROW_BAG_SIZE + player] / MAX_COINS;
            else {
                int initiative = row[ROW_INITIATIVE];
                v = (initiative < 2 ? initiative : 0) == player;
            }
        } else {
            int g = j - OFF_LOOSE - 2 * PLAYER_SCALARS;
            if (g == 0) v = row[ROW_WP];
            else if (g == 1) v = row[ROW_INIT_MOVED];
            else if (g == 2) v = row[ROW_TO_ACT] == 0;
            else {
                int k = g - 3, d = k / PENDING_KINDS;
                v = d < CONT_CAP && row[ROW_STACK_KIND + d] == k % PENDING_KINDS;
            }
        }
        dst[j] = v;
    }
}

extern "C" __global__ void k_bias(float* out, const float* b, int rows, int width) {
    int r = blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= rows) return;
    float* row = out + (size_t)r * width;
    for (int j = threadIdx.x; j < width; j += 32) row[j] += b[j];
}

static_assert(BINS <= 32, "one lane holds one value bin");

extern "C" __global__ void k_value(const float* hidden, const float* w, const float* b, float* out, int n) {
    int r = blockIdx.x * blockDim.y + threadIdx.y, lane = threadIdx.x;
    if (r >= n) return;
    const float* h = hidden + (size_t)r * HIDDEN;
    float logit = neg_inf();
    if (lane < BINS) {
        logit = b[lane];
        for (int k = 0; k < HIDDEN; ++k) logit += gelu1(h[k]) * w[k * BINS + lane];
    }
    float top = logit;
    for (int s = 16; s > 0; s >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffff, top, s));
    float e = lane < BINS ? expf(logit - top) : 0.0f;
    float centre = (lane + 0.5f) / (BINS / 2) - 1.0f;
    float total = warp_sum(e), mean = warp_sum(e * centre);
    if (lane == 0) out[r] = mean / total;
}

#define ACT_Q (TRUNK_C / 32)
static_assert(TRUNK_C % 32 == 0, "an action row splits evenly across a warp");

extern "C" __global__ void k_actions(const unsigned char* desc, const unsigned* row_of,
                                     const __half* tokens, const __half* spatial,
                                     const float* kind, const float* role, const float* pooled,
                                     const float* gamma, const float* beta, const float* w_out,
                                     float* logit, int n) {
    int a = blockIdx.x * blockDim.y + threadIdx.y, lane = threadIdx.x;
    if (a >= n) return;
    const unsigned char* d = desc + (size_t)a * ACT_BYTES;
    const unsigned row = row_of[a];
    float v[ACT_Q];
    float sum = 0.0f;
#pragma unroll
    for (int q = 0; q < ACT_Q; ++q) {
        const int j = lane + 32 * q;
        float x = kind[(size_t)d[0] * TRUNK_C + j] + pooled[(size_t)row * TRUNK_C + j];
        for (int r = 0; r < 5; ++r) {
            const int e = d[r + 1];
            const __half* src = r < 2 ? (e < NTYPE ? tokens + ((size_t)row * NTYPE + e) * TRUNK_C : nullptr)
                                      : (e < N_HEXES ? spatial + ((size_t)row * N_HEXES + e) * TRUNK_C : nullptr);
            if (src) x += role[r * TRUNK_C + j] * __half2float(src[j]);
        }
        v[q] = x;
        sum += x;
    }
    const float mean = warp_sum(sum) / TRUNK_C;
    float var = 0.0f;
#pragma unroll
    for (int q = 0; q < ACT_Q; ++q) var += (v[q] - mean) * (v[q] - mean);
    const float inv = rsqrtf(warp_sum(var) / TRUNK_C + 1e-5f);
    float acc = 0.0f;
#pragma unroll
    for (int q = 0; q < ACT_Q; ++q) {
        const int j = lane + 32 * q;
        acc += gelu1((v[q] - mean) * inv * gamma[j] + beta[j]) * w_out[j];
    }
    acc = warp_sum(acc);
    if (lane == 0) logit[a] = acc;
}

extern "C" __global__ void k_prior(const float* logit, const unsigned* off, float* out, int n) {
    int r = blockIdx.x * blockDim.y + threadIdx.y, lane = threadIdx.x;
    if (r >= n) return;
    const unsigned a = off[r], b = off[r + 1];
    float top = neg_inf();
    for (unsigned i = a + lane; i < b; i += 32) top = fmaxf(top, logit[i]);
    for (int s = 16; s > 0; s >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffff, top, s));
    float mine = 0.0f;
    for (unsigned i = a + lane; i < b; i += 32) {
        const float e = expf(logit[i] - top);
        out[i] = e;
        mine += e;
    }
    const float scale = 1.0f / warp_sum(mine);
    for (unsigned i = a + lane; i < b; i += 32) out[i] *= scale;
}
