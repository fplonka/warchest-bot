
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

extern "C" __global__ void k_narrow(const float* x, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
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


extern "C" __global__ void k_window(const float* src, float* out, int rows, int stride,
                         int off, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    out[i] = src[(size_t)(i / width) * stride + off + (i % width)];
}

extern "C" __global__ void k_tokens(const float* cards, const int* card_of_row,
                         const float* seat, float* out, int rows, int ntype,
                         int type, int nslot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * ntype * type) return;
    int r = i / (ntype * type), t = (i / type) % ntype, j = i % type;
    out[i] += cards[((size_t)card_of_row[r] * ntype + t) * type + j]
            + seat[(size_t)(t / nslot) * type + j];
}

extern "C" __global__ void k_hex_facts(const float* xpub, float* facts, int* occupant,
                            int rows, int stride, int nhex, int hex_ch,
                            int hex_facts, int ntype) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * nhex * hex_facts) return;
    int cell = i / hex_facts, j = i % hex_facts;
    const float* hex = xpub + (size_t)(cell / nhex) * stride + (size_t)(cell % nhex) * hex_ch;
    facts[i] = hex[j];
    if (j) return;
    int who = -1;
    for (int t = 0; t < ntype; ++t)
        if (hex[hex_facts + t] != 0.0f) { who = t; break; }
    occupant[cell] = who;
}

extern "C" __global__ void k_type_pool(const float* projected, float* out, int rows,
                            int ntype, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * c) return;
    int r = i / c, j = i % c;
    float acc = 0.0f;
    for (int t = 0; t < ntype; ++t)
        acc += gelu1(projected[((size_t)r * ntype + t) * c + j]);
    out[i] = acc / ntype;
}

extern "C" __global__ void k_stem(const float* facts, const float* __restrict__ w, float* out,
                       const float* projected, const int* occupant,
                       const float* bias, const float* pos, const float* glob,
                       const float* type_pool, int cells, int nhex, int ntype,
                       int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells * c) return;
    int cell = i / c, j = i % c;
    int r = cell / nhex, h = cell % nhex;
    float acc = bias[j] + pos[(size_t)h * c + j] + glob[(size_t)r * c + j]
              + type_pool[(size_t)r * c + j];
    const float* f = facts + (size_t)cell * HEX_FACTS;
    for (int k = 0; k < HEX_FACTS; ++k) acc += f[k] * w[(size_t)k * c + j];
    int t = occupant[cell];
    if (t >= 0) acc += projected[((size_t)r * ntype + t) * c + j];
    out[i] = acc;
}


#define TRUNK_Q (TRUNK_C / 32)

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


__device__ __forceinline__ void row_stats(const float* v, float* mean, float* inv) {
    float s = 0.0f;
    for (int q = 0; q < TRUNK_Q; ++q) s += v[q];
    s = warp_sum(s);
    float m = s / TRUNK_C, t = 0.0f;
    for (int q = 0; q < TRUNK_Q; ++q) {
        float d = v[q] - m;
        t += d * d;
    }
    t = warp_sum(t);
    *mean = m;
    *inv = rsqrtf(t / TRUNK_C + 1e-5f);
}

template <typename T>
__device__ __forceinline__ void board_pool(const float* ps, const float* pm, T* out) {
    extern __shared__ float part[];
    const int lane = threadIdx.x, w = threadIdx.y, nw = blockDim.y;
    for (int q = 0; q < TRUNK_Q; ++q) {
        part[(w * 2 * TRUNK_Q + q) * 32 + lane] = ps[q];
        part[(w * 2 * TRUNK_Q + TRUNK_Q + q) * 32 + lane] = pm[q];
    }
    __syncthreads();
    if (w) return;
    for (int q = 0; q < TRUNK_Q; ++q) {
        float s = 0.0f, m = neg_inf();
        for (int k = 0; k < nw; ++k) {
            s += part[(k * 2 * TRUNK_Q + q) * 32 + lane];
            m = fmaxf(m, part[(k * 2 * TRUNK_Q + TRUNK_Q + q) * 32 + lane]);
        }
        out[lane + 32 * q] = (T)(s / N_HEXES);
        out[TRUNK_C + lane + 32 * q] = (T)m;
    }
}

extern "C" __global__ void k_trunk_in(const float* __restrict__ x, __half* a, __half* pooled,
                                      const float* __restrict__ carry,
                                      const float* __restrict__ g, const float* __restrict__ b) {
    const int lane = threadIdx.x, w = threadIdx.y, row = blockIdx.x;
    float ps[TRUNK_Q], pm[TRUNK_Q];
    for (int q = 0; q < TRUNK_Q; ++q) {
        ps[q] = 0.0f;
        pm[q] = neg_inf();
    }
    for (int h = w; h < N_HEXES; h += blockDim.y) {
        size_t o = ((size_t)row * N_HEXES + h) * TRUNK_C;
        float v[TRUNK_Q];
        for (int q = 0; q < TRUNK_Q; ++q)
            v[q] = x[o + lane + 32 * q] + carry[lane + 32 * q];
        float mean, inv;
        row_stats(v, &mean, &inv);
        for (int q = 0; q < TRUNK_Q; ++q) {
            int j = lane + 32 * q;
            float y = gelu1((v[q] - mean) * inv * g[j] + b[j]);
            a[o + j] = __float2half(y);
            ps[q] += y;
            pm[q] = fmaxf(pm[q], y);
        }
    }
    board_pool(ps, pm, pooled + (size_t)row * 2 * TRUNK_C);
}

extern "C" __global__ void k_trunk_mix(const __half* __restrict__ mix,
                                       const __half* __restrict__ nbmix,
                                       const __half* __restrict__ gb,
                                       const int* __restrict__ near, const float* __restrict__ bias,
                                       const float* __restrict__ g, const float* __restrict__ b,
                                       __half* a, int cells) {
    int cell = blockIdx.x * blockDim.y + threadIdx.y;
    if (cell >= cells) return;
    const int lane = threadIdx.x;
    int row = cell / N_HEXES, h = cell - row * N_HEXES;
    const __half* here = mix + (size_t)cell * TRUNK_C;
    const __half* nb = nbmix + (size_t)row * N_HEXES * TRUNK_C;
    float v[TRUNK_Q];
    for (int q = 0; q < TRUNK_Q; ++q) {
        int j = lane + 32 * q;
        v[q] = __half2float(here[j]) + bias[j]
             + __half2float(gb[(size_t)row * TRUNK_C + j]);
    }
    for (int k = 0; k < 6; ++k) {
        int n = near[h * 6 + k];
        if (n >= 0)
            for (int q = 0; q < TRUNK_Q; ++q)
                v[q] += __half2float(nb[(size_t)n * TRUNK_C + lane + 32 * q]);
    }
    float mean, inv;
    row_stats(v, &mean, &inv);
    for (int q = 0; q < TRUNK_Q; ++q) {
        int j = lane + 32 * q;
        a[(size_t)cell * TRUNK_C + j] =
            __float2half(gelu1((v[q] - mean) * inv * g[j] + b[j]));
    }
}

extern "C" __global__ void k_trunk_out(const float* __restrict__ x, __half* spatial,
                                       const float* __restrict__ carry,
                                       const float* __restrict__ g, const float* __restrict__ b,
                                       const float* __restrict__ xpub, float* out) {
    const int lane = threadIdx.x, w = threadIdx.y, row = blockIdx.x;
    float ps[TRUNK_Q], pm[TRUNK_Q];
    for (int q = 0; q < TRUNK_Q; ++q) {
        ps[q] = 0.0f;
        pm[q] = neg_inf();
    }
    for (int h = w; h < N_HEXES; h += blockDim.y) {
        size_t o = ((size_t)row * N_HEXES + h) * TRUNK_C;
        float v[TRUNK_Q];
        for (int q = 0; q < TRUNK_Q; ++q)
            v[q] = x[o + lane + 32 * q] + carry[lane + 32 * q];
        float mean, inv;
        row_stats(v, &mean, &inv);
        for (int q = 0; q < TRUNK_Q; ++q) {
            int j = lane + 32 * q;
            float y = gelu1((v[q] - mean) * inv * g[j] + b[j]);
            spatial[o + j] = __float2half(y);
            ps[q] += y;
            pm[q] = fmaxf(pm[q], y);
        }
    }
    const size_t width = 2 * TRUNK_C + LOOSE;
    board_pool(ps, pm, out + row * width);
    if (w) return;
    for (int k = lane; k < LOOSE; k += 32)
        out[row * width + 2 * TRUNK_C + k] = xpub[(size_t)row * PUBFEAT + OFF_LOOSE + k];
}

extern "C" __global__ void k_public_cache(const Tree* trees, const unsigned int* part,
                               const unsigned int* board, __half* out, int rows,
                               int tokens, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * tokens * c) return;
    int r = i / (tokens * c), t = (i / c) % tokens, j = i % c;
    const Tree& tree = trees[part[r]];
    out[i] = t < N_HEXES
        ? tree.spatial[((size_t)board[r] * N_HEXES + t) * c + j]
        : tree.tokens[((size_t)board[r] * NTYPE + t - N_HEXES) * c + j];
}

#define ATTN_TOK (N_HEXES + NTYPE)
#define ATTN_W (ATTN / HEADS)
#define ATTN_PITCH (ATTN_W + 1)
static_assert(ATTN_W == 32, "k_attention gives one head to one warp lane by lane");
static_assert(ATTN_W % 4 == 0, "k_attention unrolls the score by four");

extern "C" __global__ void k_attention(const Tree* trees, const __half* kv,
                            const float* __restrict__ kv_b,
                            const unsigned int* part,
                            const unsigned int* board,
                            const unsigned int* config, __half* out,
                            const unsigned int* starts) {
    __shared__ float keys[ATTN_TOK * ATTN_PITCH];
    __shared__ float values[ATTN_TOK * ATTN_PITCH];
    __shared__ float query[ATTN_WARPS * ATTN_W];
    __shared__ float score[ATTN_WARPS * ATTN_TOK];
    const int head = blockIdx.y, lane = threadIdx.x, warp = threadIdx.y;
    const unsigned int begin = starts[blockIdx.x], end = starts[blockIdx.x + 1];
    const __half* board_kv =
        kv + (size_t)board[begin] * ATTN_TOK * 2 * ATTN + head * ATTN_W;
    kv_b += head * ATTN_W + lane;
    for (int t = warp; t < ATTN_TOK; t += ATTN_WARPS) {
        keys[t * ATTN_PITCH + lane] =
            __half2float(board_kv[(size_t)t * 2 * ATTN + lane]) + *kv_b;
        values[t * ATTN_PITCH + lane] =
            __half2float(board_kv[(size_t)t * 2 * ATTN + ATTN + lane]) + kv_b[ATTN];
    }
    __syncthreads();
    float* mine = query + warp * ATTN_W;
    float* sc = score + warp * ATTN_TOK;
    for (unsigned int pair = begin + warp; pair < end; pair += ATTN_WARPS) {
        const Tree& tree = trees[part[pair]];
        mine[lane] = tree.attn_query[(size_t)config[pair] * ATTN + head * ATTN_W + lane];
        __syncwarp();
        float top = neg_inf();
        for (int t = lane; t < ATTN_TOK; t += 32) {
            const float* kt = keys + t * ATTN_PITCH;
            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
            for (int e = 0; e < ATTN_W; e += 4) {
                v0 += mine[e] * kt[e];
                v1 += mine[e + 1] * kt[e + 1];
                v2 += mine[e + 2] * kt[e + 2];
                v3 += mine[e + 3] * kt[e + 3];
            }
            float v = ((v0 + v1) + (v2 + v3)) * rsqrtf((float)ATTN_W);
            sc[t] = v;
            top = fmaxf(top, v);
        }
        for (int s = 16; s > 0; s >>= 1) top = fmaxf(top, __shfl_xor_sync(0xffffffff, top, s));
        float total = 0.0f;
        for (int t = lane; t < ATTN_TOK; t += 32) {
            float v = __expf(sc[t] - top);
            sc[t] = v;
            total += v;
        }
        total = warp_sum(total);
        __syncwarp();
        float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
        int t = 0;
        for (; t + 3 < ATTN_TOK; t += 4) {
            m0 += sc[t] * values[t * ATTN_PITCH + lane];
            m1 += sc[t + 1] * values[(t + 1) * ATTN_PITCH + lane];
            m2 += sc[t + 2] * values[(t + 2) * ATTN_PITCH + lane];
            m3 += sc[t + 3] * values[(t + 3) * ATTN_PITCH + lane];
        }
        float mixed = (m0 + m1) + (m2 + m3);
        for (; t < ATTN_TOK; ++t) mixed += sc[t] * values[t * ATTN_PITCH + lane];
        out[(size_t)pair * ATTN + head * ATTN_W + lane] = __float2half(mixed / total);
        __syncwarp();
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

extern "C" __global__ void k_belief(const Tree* trees, const int* part_of_row,
                         const int* local_row, __half* out, int rows, int q0) {
    int seat = blockIdx.x * blockDim.y + threadIdx.y, lane = threadIdx.x;
    if (seat >= rows) return;
    int i = seat >> 1;
    unsigned int player = seat & 1;
    int r = q0 + i;
    const Tree& t = trees[part_of_row[r]];
    unsigned int row = local_row[r], node = t.leaf_node[row];
    unsigned int count = t.nc[2 * node + player];
    unsigned int ra = rbase(t, node, player);
    float total = 0.0f;
    for (unsigned int c = lane; c < count; c += 32) total += t.reach[ra + c];
    total = warp_sum(total);
    float inv = total > SMOOTH ? 1.0f / total : 1.0f / (float)max(count, 1u);
    unsigned int cs = t.coff[2 * row + player];
    float acc[ATTN / 32] = {0.0f};
    for (unsigned int c = 0; c < count; ++c) {
        float weight = total > SMOOTH ? t.reach[ra + c] * inv : inv;
        const __half* pp = t.conditioned + (size_t)(cs + c) * ATTN;
        for (int q = 0; q < ATTN / 32; ++q) acc[q] += weight * __half2float(pp[lane + 32 * q]);
    }
    __half* own = out + ((size_t)(2 * i) * 2 + player) * ATTN;
    __half* other = out + ((size_t)(2 * i + 1) * 2 + (1 - player)) * ATTN;
    for (int q = 0; q < ATTN / 32; ++q) {
        __half v = __float2half(acc[q]);
        own[lane + 32 * q] = v;
        other[lane + 32 * q] = v;
    }
}

__device__ __forceinline__ unsigned gelu_pair(const __half* p, float b0, float b1, bool live) {
    float2 v = __half22float2(*(const __half2*)p);
    __half2 h = __floats2half2_rn(gelu1(v.x + b0), gelu1(v.y + b1));
    return live ? *(unsigned*)&h : 0u;
}

extern "C" __global__ void k_value(const Tree* trees, const int* part_of_row,
                        const int* local_row, const float* belief_projection,
                        const unsigned* __restrict__ out_w,
                        const float* out_b, int rows, int q0) {
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int k = blockIdx.x * VALUE_WARPS + warp;
    if (k >= rows) return;
    const int r = q0 + (k >> 1);
    const unsigned int player = k & 1;
    const Tree& t = trees[part_of_row[r]];
    unsigned int row = local_row[r], node = t.leaf_node[row];
    unsigned int count = t.nc[2 * node + player];
    unsigned int cs = t.coff[2 * row + player];
    const float* __restrict__ bp = t.value_board + (size_t)t.board_of[row] * ATTN;
    const float* __restrict__ rp = belief_projection + (size_t)k * ATTN;
    const __half* __restrict__ cfg = t.value_cfg;
    float* opinion = t.opinion + player * t.nvals + t.voff[node];
    const int g = lane >> 2, tt = lane & 3;
    float base[ATTN / 16][4];
#pragma unroll
    for (int kt = 0; kt < ATTN / 16; ++kt)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int col = 16 * kt + 2 * tt + (i & 1) + 8 * (i >> 1);
            base[kt][i] = bp[col] + rp[col];
        }

    for (unsigned int c0 = 0; c0 < count; c0 += 16) {
        const bool live0 = c0 + g < count, live1 = c0 + g + 8 < count;
        const __half* p0 = cfg + (size_t)(cs + min(c0 + g, count - 1)) * ATTN + 2 * tt;
        const __half* p1 = cfg + (size_t)(cs + min(c0 + g + 8, count - 1)) * ATTN + 2 * tt;
        float d[2][4] = {};
#pragma unroll
        for (int kt = 0; kt < ATTN / 16; ++kt) {
            unsigned int af[4];
            af[0] = gelu_pair(p0 + 16 * kt, base[kt][0], base[kt][1], live0);
            af[1] = gelu_pair(p1 + 16 * kt, base[kt][0], base[kt][1], live1);
            af[2] = gelu_pair(p0 + 16 * kt + 8, base[kt][2], base[kt][3], live0);
            af[3] = gelu_pair(p1 + 16 * kt + 8, base[kt][2], base[kt][3], live1);
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
        unsigned int player = role == 0 ? me : 1 - me;
        unsigned int n = t.nc[2 * node + player], ra = rbase(t, node, player);
        float total = 0.0f;
        for (unsigned int c = lane; c < n; c += 32) total += t.reach[ra + c];
        total = warp_sum(total);
        float inv = total > SMOOTH ? 1.0f / total : 1.0f / (float)max(n, 1u);
        unsigned int cs = t.coff[2 * row_of[k] + player];
        for (int j = lane; j < attn; j += 32) {
            float acc = 0.0f;
            for (unsigned int c = 0; c < n; ++c) {
                float belief = total > SMOOTH ? t.reach[ra + c] * inv : inv;
                unsigned int pair = cs + c;
                acc += belief * __half2float(t.conditioned[(size_t)pair * attn + j]);
            }
            input[(size_t)k * (d + 2 * attn) + d + role * attn + j] = acc;
            belief_out[((size_t)k * 2 + role) * attn + j] = acc;
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

