__device__ __forceinline__ unsigned smem(const __half* p) { return (unsigned)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void load_a(const __half* p, unsigned (&f)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3]) : "r"(smem(p)));
}

struct H16 {
    unsigned r[2];
};

__device__ __forceinline__ void mma(H16& d, const unsigned (&a)[4], const unsigned (&b)[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
        : "+r"(d.r[0]), "+r"(d.r[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void cp_async16(__half* s, const __half* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem(s)), "l"(g));
}

#define WM (BM / WARPS_M)
#define WN (BN / WARPS_N)
#define THREADS (32 * WARPS_M * WARPS_N)
#define OUT (BN + 8)
#define GELU 1
#define ADD 2
static_assert(BM * OUT <= STAGES * (BM + BN) * BK, "the output tile fits in the staging buffers");

__device__ __forceinline__ int swizzle(int r, int c) { return r * BK + 8 * (c ^ ((r >> 1) & 3)); }

extern "C" __global__ void __launch_bounds__(THREADS)
k_linear(const __half* __restrict__ a, const __half* __restrict__ w, const float* __restrict__ bias, __half* y,
         int m, int n, int k, int epilogue) {
    extern __shared__ __align__(128) __half tiles[];
    const int tid = threadIdx.x, lane = tid & 31, g = lane >> 2, t = lane & 3, warp = tid >> 5;
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN, wm = WM * (warp / WARPS_N), wn = WN * (warp % WARPS_N);
    const int chunks = k / BK;
    const auto sa = [&](int c) { return tiles + c % STAGES * (BM + BN) * BK; };
    const auto load = [&](int c) {
        __half* s = sa(c);
        for (int i = tid; i < (BM + BN) * BK / 8; i += THREADS) {
            const int r = i >> 2, q = i & 3;
            const __half* src = r < BM ? a + (size_t)min(row0 + r, m - 1) * k : w + (size_t)(col0 + r - BM) * k;
            cp_async16(s + swizzle(r, q), src + c * BK + 8 * q);
        }
    };
    H16 acc[WM / 16][WN / 8] = {};
    for (int c = 0; c < STAGES - 1; ++c) {
        if (c < chunks) load(c);
        asm volatile("cp.async.commit_group;");
    }
    for (int c = 0; c < chunks; ++c) {
        asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2));
        __syncthreads();
        if (c + STAGES - 1 < chunks) load(c + STAGES - 1);
        asm volatile("cp.async.commit_group;");
        const __half* s = sa(c);
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            unsigned af[WM / 16][4], bf[WN / 8][2];
#pragma unroll
            for (int i = 0; i < WM / 16; ++i) load_a(s + swizzle(wm + 16 * i + (lane & 15), 2 * kk + (lane >> 4)), af[i]);
#pragma unroll
            for (int j = 0; j < WN / 16; ++j) {
                unsigned f[4];
                load_a(s + swizzle(BM + wn + 16 * j + (lane & 7) + 8 * (lane >> 4), 2 * kk + ((lane >> 3) & 1)), f);
                bf[2 * j][0] = f[0], bf[2 * j][1] = f[1], bf[2 * j + 1][0] = f[2], bf[2 * j + 1][1] = f[3];
            }
#pragma unroll
            for (int i = 0; i < WM / 16; ++i)
#pragma unroll
                for (int j = 0; j < WN / 8; ++j) mma(acc[i][j], af[i], bf[j]);
        }
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    __half* out = tiles;
#pragma unroll
    for (int i = 0; i < WM / 16; ++i)
#pragma unroll
        for (int j = 0; j < WN / 8; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h)
                *(unsigned*)&out[(wm + 16 * i + g + 8 * h) * OUT + wn + 8 * j + 2 * t] = acc[i][j].r[h];
    __syncthreads();
    for (int i = tid; i < BM * BN / 8; i += THREADS) {
        const int r = i / (BN / 8), c = 8 * (i % (BN / 8)), row = row0 + r;
        if (row >= m) break;
        const uint4 v = *(const uint4*)&out[r * OUT + c];
        uint4* dst = (uint4*)(y + (size_t)row * n + col0 + c);
        const uint4 x = epilogue == ADD ? *dst : make_uint4(0, 0, 0, 0);
        const __half2 *hv = (const __half2*)&v, *hx = (const __half2*)&x;
        const float4 b0 = *(const float4*)&bias[col0 + c], b1 = *(const float4*)&bias[col0 + c + 4];
        const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
        unsigned o[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            float2 f = __half22float2(hv[q]);
            f.x += b[2 * q];
            f.y += b[2 * q + 1];
            if (epilogue == GELU) f = make_float2(gelu1(f.x), gelu1(f.y));
            const float2 res = __half22float2(hx[q]);
            o[q] = pack2(f.x + res.x, f.y + res.y);
        }
        *dst = make_uint4(o[0], o[1], o[2], o[3]);
    }
}
