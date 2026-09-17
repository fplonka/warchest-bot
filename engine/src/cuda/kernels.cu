
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

__device__ __forceinline__ float2 lds2(const __half* p) {
    return word2(*(const unsigned*)p);
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

extern "C" __global__ void k_gelu(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = gelu1(x[i]);
}

static_assert(ATTN % 16 == 0, "k_value walks a configuration row in mma tiles");
static_assert(BINS == 16, "k_value writes both halves of a sixteen wide tile");

#define NORM_Q (((TRUNK_C > ATTN ? TRUNK_C : ATTN) + 31) / 32)

extern "C" __global__ void k_norm_ip(float* x, const float* gamma, const float* beta,
                          int rows, int width, int act) {
    int r = blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= rows) return;
    float* row = x + (size_t)r * width;
    float v[NORM_Q];
    int n = 0;
    float sum = 0.0f;
    for (int j = threadIdx.x; j < width; j += 32) {
        v[n++] = row[j];
        sum += row[j];
    }
    for (int s = 16; s > 0; s >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, s);
    float mean = sum / width;
    float var = 0.0f;
    for (int k = 0; k < n; ++k) {
        float d = v[k] - mean;
        var += d * d;
    }
    for (int s = 16; s > 0; s >>= 1) var += __shfl_xor_sync(0xffffffff, var, s);
    float inv = rsqrtf(var / width + 1e-5f);
    n = 0;
    for (int j = threadIdx.x; j < width; j += 32) {
        float y = (v[n++] - mean) * inv * gamma[j] + beta[j];
        row[j] = act ? gelu1(y) : y;
    }
}

extern "C" __global__ void k_bias(float* out, const float* b, int rows, int width) {
    int r = blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= rows) return;
    float* row = out + (size_t)r * width;
    for (int j = threadIdx.x; j < width; j += 32) row[j] += b[j];
}


__device__ __forceinline__ void mma_tile(float (&d)[4], const unsigned (&a)[4],
                                         const unsigned (&b)[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void frag_b(const unsigned* w, int k, int slot, int lane,
                                       int ntiles, unsigned (&f)[2]) {
    const unsigned* p = w + (((size_t)k * ntiles + slot) * 32 + lane) * 2;
    f[0] = p[0];
    f[1] = p[1];
}


#define PUBLIC_SPATIAL (N_HEXES * TRUNK_C / 8)
#define PUBLIC_TOKENS (NTYPE * TRUNK_C / 8)
static_assert(TRUNK_C % 8 == 0, "a public token row is whole 16-byte words");

extern "C" __global__ void k_public_cache(const Tree* trees, const unsigned int* part,
                               const unsigned int* board, uint4* out) {
    const int r = blockIdx.x;
    const Tree& tree = trees[part[r]];
    const uint4* spatial = (const uint4*)(tree.spatial + (size_t)board[r] * N_HEXES * TRUNK_C);
    const uint4* tokens = (const uint4*)(tree.tokens + (size_t)board[r] * NTYPE * TRUNK_C);
    uint4* dst = out + (size_t)r * (PUBLIC_SPATIAL + PUBLIC_TOKENS);
    for (int i = threadIdx.x; i < PUBLIC_SPATIAL + PUBLIC_TOKENS; i += blockDim.x)
        dst[i] = i < PUBLIC_SPATIAL ? spatial[i] : tokens[i - PUBLIC_SPATIAL];
}

#define ATTN_TOK (N_HEXES + NTYPE)
#define ATTN_W (ATTN / HEADS)
#define ATTN_KV (2 * ATTN)
#define ATTN_NT ((ATTN_TOK + 7) / 8)
#define ATTN_KT ((ATTN_TOK + 15) / 16)
#define ATTN_LDS (ATTN + 8)
static_assert(ATTN_W == 32, "k_attention gives one head two mma k-steps and four column tiles");
static_assert(ATTN % 8 == 0 && ATTN_LDS % 8 == 0, "a value row is whole 16-byte words");

__device__ __forceinline__ void frag_v(const __half* v, int kk, int col, int lane, unsigned (&f)[4]) {
    const __half* p = v + (size_t)(16 * kk + (lane & 7) + 8 * ((lane >> 3) & 1)) * ATTN_LDS + col + 8 * (lane >> 4);
    unsigned addr = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3]) : "r"(addr));
}

__device__ __forceinline__ unsigned halves(__half a, __half b) {
    __half2 h = __halves2half2(a, b);
    return *(unsigned*)&h;
}

__device__ __forceinline__ unsigned halves(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *(unsigned*)&h;
}

extern "C" __global__ void k_attention(const Tree* trees, const __half* kv,
                            const float* __restrict__ kv_b,
                            const unsigned int* part,
                            const unsigned int* board,
                            const unsigned int* config, __half* out,
                            const unsigned int* starts) {
    __shared__ __align__(16) __half sv[16 * ATTN_KT * ATTN_LDS];
    const int head = threadIdx.y, lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    const unsigned int begin = starts[blockIdx.x], end = starts[blockIdx.x + 1];
    const __half* bk = kv + (size_t)board[begin] * ATTN_TOK * ATTN_KV + head * ATTN_W;
    {
        const uint4* src = (const uint4*)(kv + (size_t)board[begin] * ATTN_TOK * ATTN_KV + ATTN);
        uint4* dst = (uint4*)sv;
        constexpr int ROW = ATTN / 8, PITCH = ATTN_LDS / 8;
        for (int i = threadIdx.y * 32 + lane; i < 16 * ATTN_KT * ROW; i += 32 * HEADS) {
            const int tok = i / ROW, j = i - tok * ROW;
            dst[tok * PITCH + j] = tok < ATTN_TOK ? src[(size_t)tok * (ATTN_KV / 8) + j] : make_uint4(0u, 0u, 0u, 0u);
        }
    }
    __syncthreads();
    const float scale = rsqrtf((float)ATTN_W);
    for (unsigned int p0 = begin; p0 < end; p0 += 16) {
        const unsigned int pa = min(p0 + g, end - 1), pb = min(p0 + g + 8, end - 1);
        const float* qa = trees[part[pa]].attn_query + (size_t)config[pa] * ATTN + head * ATTN_W;
        const float* qb = trees[part[pb]].attn_query + (size_t)config[pb] * ATTN + head * ATTN_W;
        float d[ATTN_NT][4] = {};
#pragma unroll
        for (int kk = 0; kk < ATTN_W / 16; ++kk) {
            const int c = 16 * kk + 2 * t;
            unsigned int af[4] = {
                halves(qa[c] * scale, qa[c + 1] * scale), halves(qb[c] * scale, qb[c + 1] * scale),
                halves(qa[c + 8] * scale, qa[c + 9] * scale), halves(qb[c + 8] * scale, qb[c + 9] * scale),
            };
#pragma unroll
            for (int nt = 0; nt < ATTN_NT; ++nt) {
                const int tok = 8 * nt + g;
                unsigned int bf[2] = {0u, 0u};
                if (tok < ATTN_TOK) {
                    const __half* k = bk + (size_t)tok * ATTN_KV + c;
                    bf[0] = *(const unsigned*)k;
                    bf[1] = *(const unsigned*)(k + 8);
                }
                mma_tile(d[nt], af, bf);
            }
        }
        float top[2] = {neg_inf(), neg_inf()};
#pragma unroll
        for (int nt = 0; nt < ATTN_NT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                if (8 * nt + 2 * t + (i & 1) >= ATTN_TOK) d[nt][i] = neg_inf();
                top[i >> 1] = fmaxf(top[i >> 1], d[nt][i]);
            }
        float total[2] = {0.0f, 0.0f};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            top[r] = fmaxf(top[r], __shfl_xor_sync(0xffffffff, top[r], 1));
            top[r] = fmaxf(top[r], __shfl_xor_sync(0xffffffff, top[r], 2));
        }
#pragma unroll
        for (int nt = 0; nt < ATTN_NT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                d[nt][i] = __expf(d[nt][i] - top[i >> 1]);
                total[i >> 1] += d[nt][i];
            }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            total[r] += __shfl_xor_sync(0xffffffff, total[r], 1);
            total[r] += __shfl_xor_sync(0xffffffff, total[r], 2);
        }
        float o[ATTN_W / 8][4] = {};
#pragma unroll
        for (int kk = 0; kk < ATTN_KT; ++kk) {
            unsigned int af[4] = {
                halves(d[2 * kk][0], d[2 * kk][1]), halves(d[2 * kk][2], d[2 * kk][3]),
                halves(d[2 * kk + 1][0], d[2 * kk + 1][1]), halves(d[2 * kk + 1][2], d[2 * kk + 1][3]),
            };
#pragma unroll
            for (int nt = 0; nt < ATTN_W / 8; nt += 2) {
                unsigned int bf[4];
                frag_v(sv, kk, head * ATTN_W + 8 * nt, lane, bf);
                const unsigned int b0[2] = {bf[0], bf[1]}, b1[2] = {bf[2], bf[3]};
                mma_tile(o[nt], af, b0);
                mma_tile(o[nt + 1], af, b1);
            }
        }
#pragma unroll
        for (int nt = 0; nt < ATTN_W / 8; ++nt) {
            const int dim = head * ATTN_W + 8 * nt + 2 * t;
            const float b0 = kv_b[ATTN + dim], b1 = kv_b[ATTN + dim + 1];
            if (p0 + g < end)
                *(unsigned*)(out + (size_t)(p0 + g) * ATTN + dim) =
                    halves(o[nt][0] / total[0] + b0, o[nt][1] / total[0] + b1);
            if (p0 + g + 8 < end)
                *(unsigned*)(out + (size_t)(p0 + g + 8) * ATTN + dim) =
                    halves(o[nt][2] / total[1] + b0, o[nt][3] / total[1] + b1);
        }
    }
}

extern "C" __global__ void k_attention_residual(const Tree* trees, __half* out,
                                     const unsigned int* part,
                                     const unsigned int* config,
                                     const float* gamma, const float* beta,
                                     int n) {
    const int width = ATTN;
    int r = blockIdx.x * blockDim.y + threadIdx.y;
    if (r >= n) return;
    const float* query = trees[part[r]].query + (size_t)config[r] * width;
    float values[ATTN / 32];
    float sum = 0.0f;
    int count = 0;
    for (int j = threadIdx.x; j < width; j += 32) {
        float v = __half2float(out[(size_t)r * width + j]) + query[j];
        values[count++] = v;
        sum += v;
    }
    sum = warp_sum(sum);
    float mean = sum / width, var = 0.0f;
    for (int i = 0; i < count; ++i) {
        float d = values[i] - mean;
        var += d * d;
    }
    float inv = rsqrtf(warp_sum(var) / width + 1e-5f);
    count = 0;
    for (int j = threadIdx.x; j < width; j += 32)
        out[(size_t)r * width + j] =
            __float2half((values[count++] - mean) * inv * gamma[j] + beta[j]);
}

extern "C" __global__ void k_attention_store(const Tree* trees, const unsigned int* part,
                                  const unsigned int* at, const __half* conditioned,
                                  const __half* projected, int n) {
    const int width = ATTN;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * width) return;
    int r = i / width, j = i % width;
    const Tree& tree = trees[part[r]];
    size_t to = (size_t)at[r] * width + j;
    tree.conditioned[to] = conditioned[i];
    tree.value_cfg[to] = projected[i];
}

extern "C" __global__ void k_cfg_slots(const float* phi, const unsigned int* owner,
                            const float* cards, float* slots, int n, int nslot,
                            int cfeat, int ntype, int type) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int width = 3 + type;
    if (i >= n * nslot * width) return;
    int row = i / width, j = i % width;
    int cfg = row / nslot, k = row % nslot;
    if (j == 0) slots[i] = phi[(size_t)cfg * cfeat + k];
    else if (j == 1) slots[i] = phi[(size_t)cfg * cfeat + nslot + k];
    else if (j == 2) slots[i] = phi[(size_t)cfg * cfeat + 2 * nslot + k];
    else slots[i] = cards[((size_t)owner[cfg] * nslot + k) * type + j - 3];
}

extern "C" __global__ void k_cfg_seat(float* query, const unsigned int* owner,
                           const float* seat, int n, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * width) return;
    query[i] += seat[(size_t)(owner[i / width] & 1u) * width + i % width];
}

extern "C" __global__ void k_sum_slots(const float* hidden, float* out, int n, int nslot,
                            int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * width) return;
    int cfg = i / width, j = i % width;
    float acc = 0.0f;
    for (int k = 0; k < nslot; ++k)
        acc += hidden[((size_t)cfg * nslot + k) * width + j];
    out[i] = acc;
}




extern "C" __global__ void k_scatter(const unsigned int* blob, unsigned int* const* dst,
                          const unsigned int* at, const unsigned int* src,
                          const unsigned int* sum, int pieces, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int lo = 0, hi = pieces;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if ((int)sum[mid] <= i) lo = mid; else hi = mid;
    }
    int k = i - (int)sum[lo];
    dst[lo][at[lo] + k] = blob[src[lo] + k];
}


__device__ __forceinline__ float cfr_factor(float t, float p) {
    if (isinf(p)) return p > 0.0f ? 1.0f : 0.0f;
    float x = powf(t, p);
    return x / (x + 1.0f);
}

#define SMOOTH 1e-30f
#define NO_ROW 0xffffffffu
#define NO_TRANS 0xffffffffu
#define KIND_LEAF 2u
#define KIND_CHANCE 1u

__device__ __forceinline__ unsigned int rbase(const Tree& t, unsigned int i, int p) {
    return t.roff[i] + (p == 1 ? t.nc[2 * i] : 0);
}

#define BELIEF_LANES (ATTN / 8)
#define VALUE_TILE 16
#define BELIEF_FAT 64
static_assert(BELIEF_LANES <= 32 && ATTN % 8 == 0, "a belief row is one 16-byte word a lane");

struct Seat {
    unsigned count;
    unsigned ra;
    unsigned cs;
    float inv;
};

__device__ __forceinline__ Seat seat_of(const Tree& t, unsigned row, unsigned player, int lane) {
    const unsigned node = t.leaf_node[row];
    Seat seat = { t.nc[2 * node + player], rbase(t, node, player), t.coff[2 * row + player], 0.0f };
    float total = 0.0f;
    for (unsigned c = lane; c < seat.count; c += 32) total += t.reach[seat.ra + c];
    total = warp_sum(total);
    seat.inv = total > SMOOTH ? 1.0f / total : 0.0f;
    return seat;
}

__device__ __forceinline__ float seat_weight(const Tree& t, const Seat& seat, unsigned c) {
    return seat.inv > 0.0f ? t.reach[seat.ra + c] * seat.inv : 1.0f / (float)max(seat.count, 1u);
}

__device__ __forceinline__ void belief_sum(const Tree& t, const Seat& seat, unsigned from, unsigned to,
                                           int lane, float (&acc)[8]) {
    const uint4* pp = (const uint4*)(t.conditioned + (size_t)seat.cs * ATTN) + lane;
#pragma unroll 4
    for (unsigned c = from; c < to; ++c) {
        const float weight = seat_weight(t, seat, c);
        if (weight == 0.0f || lane >= BELIEF_LANES) continue;
        const uint4 w = pp[(size_t)c * (ATTN / 8)];
        const unsigned words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float2 v = word2(words[q]);
            acc[2 * q] += weight * v.x;
            acc[2 * q + 1] += weight * v.y;
        }
    }
}

__device__ __forceinline__ void belief_store(__half* out, int i, unsigned player, int lane,
                                             const float (&acc)[8]) {
    if (lane >= BELIEF_LANES) return;
    const uint4 v = make_uint4(pack2(acc[0], acc[1]), pack2(acc[2], acc[3]),
                               pack2(acc[4], acc[5]), pack2(acc[6], acc[7]));
    ((uint4*)(out + ((size_t)(2 * i) * 2 + player) * ATTN))[lane] = v;
    ((uint4*)(out + ((size_t)(2 * i + 1) * 2 + (1 - player)) * ATTN))[lane] = v;
}

extern "C" __global__ void __launch_bounds__(32 * VALUE_WARPS)
k_belief(const Tree* trees, const int* part_of_row, const int* local_row, __half* out, int rows, int q0) {
    __shared__ unsigned counts[VALUE_WARPS];
    __shared__ float part[VALUE_WARPS][ATTN];
    const int lane = threadIdx.x, warp = threadIdx.y, k0 = blockIdx.x * VALUE_WARPS;
    const int mine = k0 + warp;
    Seat seat = {};
    if (mine < rows) {
        const Tree& t = trees[part_of_row[q0 + (mine >> 1)]];
        seat = seat_of(t, local_row[q0 + (mine >> 1)], mine & 1, lane);
    }
    if (lane == 0) counts[warp] = seat.count;
    __syncthreads();
    for (int w = 0; w < VALUE_WARPS; ++w) {
        if (counts[w] <= BELIEF_FAT) continue;
        const int k = k0 + w;
        const Tree& t = trees[part_of_row[q0 + (k >> 1)]];
        const Seat fat = seat_of(t, local_row[q0 + (k >> 1)], k & 1, lane);
        float acc[8] = {};
        belief_sum(t, fat, fat.count * warp / VALUE_WARPS, fat.count * (warp + 1) / VALUE_WARPS, lane, acc);
        if (lane < BELIEF_LANES)
#pragma unroll
            for (int q = 0; q < 8; ++q) part[warp][8 * lane + q] = acc[q];
        __syncthreads();
        if (warp == 0) {
            float sum[8] = {};
#pragma unroll
            for (int v = 0; v < VALUE_WARPS; ++v)
#pragma unroll
                for (int q = 0; q < 8; ++q) sum[q] += lane < BELIEF_LANES ? part[v][8 * lane + q] : 0.0f;
            belief_store(out, k >> 1, k & 1, lane, sum);
        }
        __syncthreads();
    }
    if (mine >= rows || seat.count > BELIEF_FAT) return;
    const Tree& t = trees[part_of_row[q0 + (mine >> 1)]];
    float acc[8] = {};
    belief_sum(t, seat, 0, seat.count, lane, acc);
    belief_store(out, mine >> 1, mine & 1, lane, acc);
}

__device__ __forceinline__ unsigned gelu_pair(unsigned raw, const float* __restrict__ bp,
                                              const float* __restrict__ rp, int col, bool live) {
    const float2 v = word2(raw);
    return live ? pack2(gelu1(v.x + bp[col] + rp[col]), gelu1(v.y + bp[col + 1] + rp[col + 1])) : 0u;
}

__device__ void value_tile(const Tree& t, unsigned row, unsigned player, unsigned c0,
                           const float* __restrict__ rp, const unsigned* __restrict__ out_w,
                           const float* out_b, int lane) {
    const unsigned node = t.leaf_node[row];
    const unsigned count = t.nc[2 * node + player], cs = t.coff[2 * row + player];
    const float* __restrict__ bp = t.value_board + (size_t)t.board_of[row] * ATTN;
    const __half* __restrict__ cfg = t.value_cfg;
    float* opinion = t.opinion + player * t.nvals + t.voff[node];
    const int g = lane >> 2, tt = lane & 3;
    const bool live0 = c0 + g < count, live1 = c0 + g + 8 < count;
    const unsigned* p0 = (const unsigned*)(cfg + (size_t)(cs + min(c0 + g, count - 1)) * ATTN) + tt;
    const unsigned* p1 = (const unsigned*)(cfg + (size_t)(cs + min(c0 + g + 8, count - 1)) * ATTN) + tt;
    unsigned raw[ATTN / 16][4];
#pragma unroll
    for (int kt = 0; kt < ATTN / 16; ++kt) {
        raw[kt][0] = p0[8 * kt];
        raw[kt][1] = p1[8 * kt];
        raw[kt][2] = p0[8 * kt + 4];
        raw[kt][3] = p1[8 * kt + 4];
    }
    float d[2][4] = {};
#pragma unroll
    for (int kt = 0; kt < ATTN / 16; ++kt) {
        const int col = 16 * kt + 2 * tt;
        unsigned int af[4];
        af[0] = gelu_pair(raw[kt][0], bp, rp, col, live0);
        af[1] = gelu_pair(raw[kt][1], bp, rp, col, live1);
        af[2] = gelu_pair(raw[kt][2], bp, rp, col + 8, live0);
        af[3] = gelu_pair(raw[kt][3], bp, rp, col + 8, live1);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            unsigned int bf[2];
            frag_b(out_w, kt, nt, lane, 2, bf);
            mma_tile(d[nt], af, bf);
        }
    }
    for (int part = 0; part < 2; ++part) {
        float v[4] = { d[0][2 * part], d[0][2 * part + 1],
                       d[1][2 * part], d[1][2 * part + 1] };
        const int b0 = 2 * tt;
        float top = -1e30f;
        for (int i = 0; i < 4; ++i) {
            v[i] += out_b[b0 + (i & 1) + 8 * (i >> 1)];
            top = fmaxf(top, v[i]);
        }
        for (int m = 1; m < 4; m <<= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffff, top, m));
        float norm = 0.0f, mean = 0.0f;
        for (int i = 0; i < 4; ++i) {
            int bin = b0 + (i & 1) + 8 * (i >> 1);
            float e = __expf(v[i] - top);
            norm += e;
            mean += e * ((bin + 0.5f) * (2.0f / BINS) - 1.0f);
        }
        for (int m = 1; m < 4; m <<= 1) {
            norm += __shfl_xor_sync(0xffffffff, norm, m);
            mean += __shfl_xor_sync(0xffffffff, mean, m);
        }
        unsigned int c = c0 + g + 8 * part;
        if (tt == 0 && c < count) opinion[c] = mean / norm;
    }
}

extern "C" __global__ void __launch_bounds__(32 * VALUE_WARPS)
k_value(const Tree* trees, const int* part_of_row, const int* local_row,
        const float* belief_projection, const unsigned* __restrict__ out_w,
        const float* out_b, int rows, int q0) {
    __shared__ unsigned tiles[VALUE_WARPS + 1];
    const int lane = threadIdx.x, warp = threadIdx.y, k0 = blockIdx.x * VALUE_WARPS;
    if (lane == 0) {
        const int k = k0 + warp;
        unsigned n = 0;
        if (k < rows) {
            const Tree& t = trees[part_of_row[q0 + (k >> 1)]];
            n = (t.nc[2 * t.leaf_node[local_row[q0 + (k >> 1)]] + (k & 1)] + VALUE_TILE - 1) / VALUE_TILE;
        }
        tiles[warp + 1] = n;
    }
    __syncthreads();
    if (threadIdx.x == 0 && warp == 0) {
        tiles[0] = 0;
        for (int w = 0; w < VALUE_WARPS; ++w) tiles[w + 1] += tiles[w];
    }
    __syncthreads();
    for (unsigned tile = warp; tile < tiles[VALUE_WARPS]; tile += VALUE_WARPS) {
        int w = 0;
        while (tiles[w + 1] <= tile) ++w;
        const int k = k0 + w;
        const Tree& t = trees[part_of_row[q0 + (k >> 1)]];
        value_tile(t, local_row[q0 + (k >> 1)], k & 1, (tile - tiles[w]) * VALUE_TILE,
                   belief_projection + (size_t)k * ATTN, out_w, out_b, lane);
    }
}

extern "C" __global__ void k_act_feats(const Tree* trees, const unsigned int* part,
                            const unsigned int* row_of, const unsigned int* desc,
                            const unsigned int* node_of, const float* kind,
                            const float* role, float* out, int n, int ntype,
                            int nhex, int chan) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * chan) return;
    int action = i / chan, j = i % chan;
    const unsigned int* d = desc + 6 * action;
    int node = node_of[action];
    const Tree& t = trees[part[node]];
    unsigned int board = t.board_of[row_of[node]];
    float v = kind[(size_t)d[0] * chan + j];
    for (int r = 0; r < 5; ++r) {
        unsigned int entity = d[r + 1];
        const __half* src = nullptr;
        if (r < 2 && entity < (unsigned int)ntype)
            src = t.tokens + ((size_t)board * ntype + entity) * chan;
        if (r >= 2 && entity < (unsigned int)nhex)
            src = t.spatial + ((size_t)board * nhex + entity) * chan;
        if (src) v += role[r * chan + j] * (float)src[j];
    }
    out[i] = v;
}

extern "C" __global__ void k_policy_inputs(const Tree* trees, const unsigned int* part,
                                const unsigned int* node_of,
                                const unsigned int* row_of, float* input,
                                float* belief_out, float* board_out, int m,
                                int attn, int d) {
    int k = blockIdx.x, lane = threadIdx.x;
    if (k >= m) return;
    const Tree& t = trees[part[k]];
    unsigned int node = node_of[k], me = t.player[node];
    for (int role = 0; role < 2; ++role) {
        const Seat seat = seat_of(t, row_of[k], role == 0 ? me : 1 - me, lane);
        float acc[8] = {};
        belief_sum(t, seat, 0, seat.count, lane, acc);
        if (lane < BELIEF_LANES)
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const int col = 8 * lane + q;
                input[(size_t)k * (d + 2 * attn) + d + role * attn + col] = acc[q];
                belief_out[((size_t)k * 2 + role) * attn + col] = acc[q];
            }
    }
    unsigned int board = t.board_of[row_of[k]];
    for (int j = lane; j < d; j += 32) {
        float v = t.p[(size_t)board * d + j];
        input[(size_t)k * (d + 2 * attn) + j] = v;
        board_out[(size_t)k * d + j] = v;
    }
}

extern "C" __global__ void k_act_add(float* z, const float* proj, const unsigned int* of,
                          int n, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * width) return;
    z[i] += proj[(size_t)of[i / width] * width + i % width];
}

extern "C" __global__ void k_policy_offset(const float* action, const float* belief,
                                const float* bias, const unsigned int* node,
                                float* out, int n, int d) {
    int a = blockIdx.x * blockDim.y + threadIdx.y, lane = threadIdx.x;
    if (a >= n) return;
    float acc = 0.0f;
    for (int j = lane; j < d; j += 32)
        acc += action[(size_t)a * d + j] * (belief[(size_t)node[a] * d + j] + bias[j]);
    acc = warp_sum(acc);
    if (lane == 0) out[a] = acc;
}

extern "C" __global__ void k_prior(const Tree* trees, const unsigned int* part,
                        const unsigned int* node_of, const unsigned int* row_of,
                        const unsigned int* act_at, const unsigned int* cell_at,
                        const unsigned int* cells, const float* e,
                        const float* offset,
                        const float* inv_t, int m) {
    const int d = ATTN;
    int k = blockIdx.y;
    if (k >= m) return;
    int c = blockIdx.x * blockDim.y + threadIdx.y;
    const Tree& t = trees[part[k]];
    unsigned int node = node_of[k];
    unsigned int me = t.player[node];
    if ((unsigned)c >= t.nc[2 * node + me]) return;
    unsigned int lb = t.legal_base[node], so = t.soff[node];
    unsigned int a = t.legal_off[lb + c], b = t.legal_off[lb + c + 1];
    if (a == b) return;
    int lane = threadIdx.x;
    unsigned int cs = t.coff[2 * row_of[k] + me];
    unsigned int pair = cs + t.cell_row[so + a];
    const __half* pp = t.conditioned + (size_t)pair * d;
    float fp[ATTN / 32];
    for (int q = 0; q < d / 32; ++q) fp[q] = __half2float(pp[lane + 32 * q]);
    for (unsigned int cell = a; cell < b; ++cell) {
        unsigned int action = act_at[k] + cells[cell_at[k] + cell];
        const float* ea = e + (size_t)action * d;
        float acc = 0.0f;
        for (int q = 0; q < d / 32; ++q) acc += fp[q] * ea[lane + 32 * q];
        acc = warp_sum(acc);
        if (lane == 0) t.prior[so + cell] = acc + offset[action];
    }
    __syncwarp();
    float top = neg_inf();
    for (unsigned int cell = a + lane; cell < b; cell += 32)
        top = fmaxf(top, t.prior[so + cell]);
    for (int s = 16; s > 0; s >>= 1)
        top = fmaxf(top, __shfl_xor_sync(0xffffffff, top, s));
    float mine = 0.0f;
    for (unsigned int cell = a + lane; cell < b; cell += 32) {
        float v = expf((t.prior[so + cell] - top) * inv_t[k]);
        t.prior[so + cell] = v;
        mine += v;
    }
    float total = warp_sum(mine);
    float scale = total > SMOOTH ? 1.0f / total : 1.0f / (float)(b - a);
    for (unsigned int cell = a + lane; cell < b; cell += 32) t.prior[so + cell] *= scale;
}

