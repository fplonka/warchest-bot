#define DH (D / HEADS)
#define LD (DH + 8)
static_assert(TOKENS % 16 == 0 && DH % 16 == 0 && D % 64 == 0, "attention runs on whole mma tiles");

struct F32 {
    float r[4];
};

__device__ __forceinline__ void mma(F32& d, const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d.r[0]), "+f"(d.r[1]), "+f"(d.r[2]), "+f"(d.r[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void load_b(const __half* p, unsigned (&f)[2]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(f[0]), "=r"(f[1]) : "r"(smem(p)));
}

__device__ __forceinline__ void load_bt(const __half* p, unsigned (&f)[2]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(f[0]), "=r"(f[1]) : "r"(smem(p)));
}

struct Stem {
    const float* card;
    const float* pile;
    const float* seat;
    const float* hex_w;
    const float* hex_b;
    const float* pos;
    const float* state_w;
    const float* state_b;
};

extern "C" __global__ void __launch_bounds__(D)
k_stem(const float* __restrict__ xpub, const unsigned char* __restrict__ rows, const Stem st, __half* x, int n) {
    __shared__ float in[PUBFEAT];
    const int b = blockIdx.x, c = threadIdx.x;
    for (int i = c; i < PUBFEAT; i += D) in[i] = xpub[(size_t)b * PUBFEAT + i];
    __syncthreads();
    __half* out = x + (size_t)b * D + c;
    float coin[NTYPE];
#pragma unroll
    for (int t = 0; t < NTYPE; ++t) {
        float v = st.card[rows[(size_t)b * ROW_BYTES + ROW_IDS + t] * D + c] + st.seat[t / NSLOT * D + c];
#pragma unroll
        for (int k = 0; k < PILE_COUNTS; ++k) v += in[OFF_PILES + t * PILE_COUNTS + k] * st.pile[c * PILE_COUNTS + k];
        coin[t] = v;
        out[(size_t)(N_HEXES + t) * n * D] = __float2half(v);
    }
    for (int h = 0; h < N_HEXES; ++h) {
        const float* f = in + h * HEX_CH;
        float v = st.hex_b[c] + st.pos[h * D + c];
        for (int k = 0; k < HEX_FACTS; ++k)
            if (f[k] != 0.0f) v += f[k] * st.hex_w[c * HEX_FACTS + k];
#pragma unroll
        for (int t = 0; t < NTYPE; ++t) v += f[HEX_FACTS + t] * coin[t];
        out[(size_t)h * n * D] = __float2half(v);
    }
    float v = st.state_b[c];
    for (int k = 0; k < LOOSE; ++k)
        if (in[OFF_LOOSE + k] != 0.0f) v += in[OFF_LOOSE + k] * st.state_w[c * LOOSE + k];
    out[(size_t)STATE * n * D] = __float2half(v);
}

extern "C" __global__ void __launch_bounds__(256)
k_norm(const __half* __restrict__ x, const float* __restrict__ ln, __half* y, int rows) {
    const int r = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= rows) return;
    const __half2* src = (const __half2*)(x + (size_t)r * D);
    float2 v[D / 64];
    float sum = 0.0f;
#pragma unroll
    for (int q = 0; q < D / 64; ++q) {
        v[q] = __half22float2(src[lane + 32 * q]);
        sum += v[q].x + v[q].y;
    }
    const float mean = warp_sum(sum) / D;
    float var = 0.0f;
#pragma unroll
    for (int q = 0; q < D / 64; ++q) var += (v[q].x - mean) * (v[q].x - mean) + (v[q].y - mean) * (v[q].y - mean);
    const float inv = rsqrtf(warp_sum(var) / D + 1e-5f);
    unsigned* dst = (unsigned*)(y + (size_t)r * D);
#pragma unroll
    for (int q = 0; q < D / 64; ++q) {
        const int c = 2 * (lane + 32 * q);
        dst[c / 2] = pack2((v[q].x - mean) * inv * ln[c] + ln[D + c], (v[q].y - mean) * inv * ln[c + 1] + ln[D + c + 1]);
    }
}

extern "C" __global__ void __launch_bounds__(2 * TOKENS)
k_attend(const __half* __restrict__ qkv, const float* __restrict__ bias, __half* out, int n) {
    __shared__ __align__(16) __half s[3][TOKENS][LD];
    const int row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int q0 = 16 * (threadIdx.x >> 5);
    const __half* src = qkv + (size_t)row * 3 * D + head * DH;
#pragma unroll
    for (int i = threadIdx.x; i < 3 * TOKENS * DH / 8; i += 2 * TOKENS) {
        const int m = i / (TOKENS * DH / 8), tok = i / (DH / 8) % TOKENS, c = 8 * (i % (DH / 8));
        *(uint4*)&s[m][tok][c] = *(const uint4*)(src + (size_t)tok * n * 3 * D + m * D + c);
    }
    __syncthreads();
    F32 p[TOKENS / 8] = {};
    for (int k = 0; k < DH; k += 16) {
        unsigned a[4];
        load_a(&s[0][q0 + (lane & 15)][k + 8 * (lane >> 4)], a);
#pragma unroll
        for (int j = 0; j < TOKENS / 8; ++j) {
            unsigned b[2];
            load_b(&s[1][8 * j + (lane & 7)][k + 8 * ((lane >> 3) & 1)], b);
            mma(p[j], a, b);
        }
    }
    const float scale = rsqrtf(DH), *bt = bias + (size_t)head * TOKENS * TOKENS + (q0 + g) * TOKENS + 2 * t;
    float top[2] = {neg_inf(), neg_inf()}, sum[2] = {0.0f, 0.0f};
#pragma unroll
    for (int j = 0; j < TOKENS / 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            p[j].r[e] = p[j].r[e] * scale + bt[(e >> 1) * 8 * TOKENS + 8 * j + (e & 1)];
            top[e >> 1] = fmaxf(top[e >> 1], p[j].r[e]);
        }
#pragma unroll
    for (int h = 0; h < 2; ++h)
        for (int o = 1; o < 4; o <<= 1) top[h] = fmaxf(top[h], __shfl_xor_sync(0xffffffff, top[h], o));
#pragma unroll
    for (int j = 0; j < TOKENS / 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            p[j].r[e] = __expf(p[j].r[e] - top[e >> 1]);
            sum[e >> 1] += p[j].r[e];
        }
#pragma unroll
    for (int h = 0; h < 2; ++h)
        for (int o = 1; o < 4; o <<= 1) sum[h] += __shfl_xor_sync(0xffffffff, sum[h], o);
    F32 o[DH / 8] = {};
#pragma unroll
    for (int k = 0; k < TOKENS / 16; ++k) {
        const F32 &lo = p[2 * k], &hi = p[2 * k + 1];
        const unsigned a[4] = {pack2(lo.r[0], lo.r[1]), pack2(lo.r[2], lo.r[3]), pack2(hi.r[0], hi.r[1]), pack2(hi.r[2], hi.r[3])};
#pragma unroll
        for (int j = 0; j < DH / 8; ++j) {
            unsigned b[2];
            load_bt(&s[2][16 * k + (lane & 15)][8 * j], b);
            mma(o[j], a, b);
        }
    }
    __half* dst = out + ((size_t)(q0 + g) * n + row) * D + head * DH + 2 * t;
#pragma unroll
    for (int j = 0; j < DH / 8; ++j) {
        *(unsigned*)(dst + 8 * j) = pack2(o[j].r[0] / sum[0], o[j].r[1] / sum[0]);
        *(unsigned*)(dst + (size_t)8 * n * D + 8 * j) = pack2(o[j].r[2] / sum[1], o[j].r[3] / sum[1]);
    }
}

struct Head {
    const float* kind;
    const float* none;
    const float* value_w;
    const float* value_b;
    const float* move_w;
    const __half* coins;
    const __half* hexes;
    const __half* state;
};

extern "C" __global__ void __launch_bounds__(256)
k_heads(const Head h, const unsigned char* __restrict__ desc, const unsigned* __restrict__ off, float* out, int n) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int first = blockIdx.x * HR, rows = min(n - first, HR);
    for (int r = warp; r < rows; r += 8) {
        const __half* v = h.state + (size_t)(first + r) * 2 * D + D;
        float acc = 0.0f;
        for (int c = lane; c < D; c += 32) acc += gelu1(__half2float(v[c])) * h.value_w[c];
        acc = warp_sum(acc);
        if (lane == 0) out[first + r] = tanhf(0.5f * (acc + h.value_b[0]));
    }
    float* prior = out + n;
    for (unsigned i = off[first] + warp; i < off[first + rows]; i += 8) {
        int r = 0;
        while (off[first + r + 1] <= i) ++r;
        const int row = first + r;
        const unsigned char* d = desc + (size_t)i * ACT_BYTES;
        const __half* pick[ACT_BYTES - 1];
#pragma unroll
        for (int s = 0; s < ACT_BYTES - 1; ++s) {
            const int coin = s < 2, slot = coin ? s : s - 2, width = coin ? 2 * D : 3 * D, at = d[1 + s];
            pick[s] = at < (coin ? NTYPE : N_HEXES) ? (coin ? h.coins : h.hexes) + ((size_t)at * n + row) * width + slot * D : nullptr;
        }
        const __half* state = h.state + (size_t)row * 2 * D;
        float acc = 0.0f;
        for (int c = lane; c < D; c += 32) {
            float v = h.kind[d[0] * D + c] + __half2float(state[c]);
#pragma unroll
            for (int s = 0; s < ACT_BYTES - 1; ++s) v += pick[s] ? __half2float(pick[s][c]) : h.none[s * D + c];
            acc += gelu1(v) * h.move_w[c];
        }
        acc = warp_sum(acc);
        if (lane == 0) prior[i] = acc;
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
