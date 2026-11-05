#include <metal_stdlib>
using namespace metal;

#define DH (D / HEADS)
#define LDP (D + 8)
static_assert(D == 256 && TOKENS % 16 == 0, "the kernels assume these shapes");

constant int EPILOGUE [[function_constant(0)]];

inline float gelu(float x) {
    const float u = 0.7978845608f * (x + 0.044715f * x * x * x);
    return x - x / (exp(2.0f * u) + 1.0f);
}

inline ushort2 owned(ushort lane) {
    const ushort q = lane / 4;
    return ushort2((q & 2) * 2 + (lane % 2) * 2, (q & 4) + (lane / 2) % 4);
}

inline void deposit(threadgroup float* to, float x, ushort lane) {
    const float t = x + simd_shuffle_xor(x, 1), u = t + simd_shuffle_xor(t, 8);
    if ((lane & 9) == 0) *to = u;
}

inline float total(threadgroup const float* blocks) {
    float s = 0.0f;
    for (ushort g = 0; g < D / 32; ++g) s += blocks[g];
    return s;
}

inline float coin(device const uchar* row, ushort t, ushort c, device const float* card, device const float* seat,
                  device const float* pile) {
    float v = card[row[ROW_IDS + t] * D + c] + seat[t / NSLOT * D + c];
    for (ushort k = 0; k < PILE_COUNTS; ++k) v += row[ROW_PILES + t * PILE_COUNTS + k] / 5.0f * pile[k * D + c];
    return v;
}

kernel void stem(device const uchar* rows [[buffer(0)]], device const float* card [[buffer(1)]],
                 device const float* pile [[buffer(2)]], device const float* seat [[buffer(3)]],
                 device const float* hex_w [[buffer(4)]], device const float* hex_b [[buffer(5)]],
                 device const float* pos [[buffer(6)]], device const float* state_w [[buffer(7)]],
                 device const float* state_b [[buffer(8)]], device const uchar* locations [[buffer(9)]],
                 device half* x [[buffer(10)]], device half* a [[buffer(11)]], device const float* ln [[buffer(12)]],
                 constant uint& n [[buffer(13)]], uint2 tg [[threadgroup_position_in_grid]],
                 ushort c [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                 ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[2][D / 32];
    const uint b = tg.x;
    const ushort tok = tg.y;
    device const uchar* row = rows + (ulong)b * ROW_BYTES;
    float v;
    if (tok < N_HEXES) {
        const uchar owner = row[ROW_HEX_OWNER + tok], slot = row[ROW_HEX_SLOT + tok], marker = row[ROW_HEX_MARKER + tok];
        v = hex_b[c] + pos[tok * D + c];
        if (owner < 2) {
            v += hex_w[owner * D + c] + row[ROW_HEX_HEIGHT + tok] / 5.0f * hex_w[2 * D + c];
            if (slot < NSLOT) v += coin(row, owner * NSLOT + slot, c, card, seat, pile);
        }
        if (marker < 2) v += hex_w[(3 + marker) * D + c];
        if (locations[tok]) v += hex_w[5 * D + c];
        for (ushort d = 0; d < CONT_CAP; ++d)
            if ((row[ROW_STACK_OWED + 8 * d + tok / 8] >> (tok % 8)) & 1) v += hex_w[(6 + d) * D + c];
    } else if (tok < STATE) {
        v = coin(row, tok - N_HEXES, c, card, seat, pile);
    } else {
        v = state_b[c];
        ushort on_board[2] = {0, 0};
        for (ushort h = 0; h < N_HEXES; ++h)
            if (row[ROW_HEX_MARKER + h] < 2) ++on_board[row[ROW_HEX_MARKER + h]];
        const uchar initiative = row[ROW_INITIATIVE] < 2 ? row[ROW_INITIATIVE] : 0;
        for (ushort p = 0; p < 2; ++p) {
            const float f[PLAYER_SCALARS] = {(6 - on_board[p]) / 6.0f, on_board[p] / 6.0f, row[ROW_HAND_SIZE + p] / 3.0f,
                                             row[ROW_FD_SIZE + p] / MAX_COINS, row[ROW_BAG_SIZE + p] / MAX_COINS,
                                             float(initiative == p)};
            for (ushort k = 0; k < PLAYER_SCALARS; ++k) v += f[k] * state_w[(p * PLAYER_SCALARS + k) * D + c];
        }
        device const float* g = state_w + 2 * PLAYER_SCALARS * D + c;
        v += row[ROW_WP] * g[0] + row[ROW_INIT_MOVED] * g[D] + float(row[ROW_TO_ACT] == 0) * g[2 * D];
        for (ushort d = 0; d < CONT_CAP; ++d)
            if (row[ROW_STACK_KIND + d] < PENDING_KINDS) v += g[(3 + d * PENDING_KINDS + row[ROW_STACK_KIND + d]) * D];
    }
    const float s = simd_sum(v);
    if (lane == 0) part[0][sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float d = v - total(part[0]) / D, q = simd_sum(d * d);
    if (lane == 0) part[1][sg] = q;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong at = ((ulong)tok * n + b) * D + c;
    x[at] = half(v);
    a[at] = half(d * rsqrt(total(part[1]) / D + 1e-5f) * ln[c] + ln[D + c]);
}

inline void put(device half* y, uint row, uint col, uint m, uint n, float2 v) {
    if (EPILOGUE != SPLIT) {
        *(device half2*)&y[(ulong)row * n + col] = half2(v);
        return;
    }
    const uint positions = m / TOKENS, tok = row / positions;
    const ulong block = ((ulong)(row - tok * positions) * HEADS + col % D / DH) * 3 + col / D;
    *(device half2*)&y[block * TOKENS * DH + tok * DH + col % DH] = half2(v);
}

template <int PM, int PSG, int PTN>
void panel(device const half* a, device const half* w, device const float* bias, device half* y, device half* normed,
           device const float* ln, uint3 dims, uint2 tg, ushort tid, ushort sg, ushort lane, threadgroup half* rows,
           threadgroup float (*part)[PM][D / 32]) {
    const uint m = dims.x, n = dims.y, k = dims.z;
    const uint row0 = tg.y * PM, nb0 = (tg.x * PSG + sg) * PTN;
    simdgroup_float8x8 acc[PM / 8][PTN];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < PM / 8; ++i)
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < PTN; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < k; k0 += D) {
        if (k0) threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort i = tid; i < PM * D / 8; i += 32 * PSG) {
            const ushort r = i / (D / 8), q = i % (D / 8);
            *(threadgroup uint4*)&rows[r * LDP + 8 * q] = *(device const uint4*)&a[(ulong)min(row0 + r, m - 1) * k + k0 + 8 * q];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const half* wb = w + ((ulong)k0 / 8 * (n / 8) + nb0) * 64;
        simdgroup_half8x8 bf[PTN], next[PTN];
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < PTN; ++j) simdgroup_load(bf[j], wb + 64 * j, 8);
        for (ushort kk = 0; kk < D; kk += 8) {
            wb += 8 * n;
            if (kk + 8 < D)
                #pragma clang loop unroll(full)
                for (ushort j = 0; j < PTN; ++j) simdgroup_load(next[j], wb + 64 * j, 8);
            simdgroup_half8x8 af[PM / 8];
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < PM / 8; ++i) simdgroup_load(af[i], &rows[8 * i * LDP + kk], LDP);
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < PM / 8; ++i)
                #pragma clang loop unroll(full)
                for (ushort j = 0; j < PTN; ++j) simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
            #pragma clang loop unroll(full)
            for (ushort j = 0; j < PTN; ++j) bf[j] = next[j];
        }
    }
    const ushort2 f = owned(lane);
    if (EPILOGUE != ADD) {
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < PM / 8; ++i) {
            const uint row = row0 + 8 * i + f.y;
            if (row >= m) continue;
            #pragma clang loop unroll(full)
            for (ushort j = 0; j < PTN; ++j) {
                const uint col = (nb0 + j) * 8 + f.x;
                const thread auto& e = acc[i][j].thread_elements();
                float2 v = float2(e[0], e[1]) + float2(bias[col], bias[col + 1]);
                if (EPILOGUE == GELU) v = float2(gelu(v.x), gelu(v.y));
                put(y, row, col, m, n, v);
            }
        }
        return;
    }
    static_assert(8 * PSG * PTN == D && PTN % 4 == 0, "a residual tile spans whole rows in 32-column blocks");
    float2 v[PM / 8][PTN];
    float mean[PM / 8];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < PM / 8; ++i) {
        const uint row = min(row0 + 8 * i + f.y, m - 1);
        float s[PTN / 4] = {};
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < PTN; ++j) {
            const ushort col = (nb0 + j) * 8 + f.x;
            const thread auto& e = acc[i][j].thread_elements();
            v[i][j] = float2(e[0], e[1]) + float2(bias[col], bias[col + 1]) + float2(*(device const half2*)&y[(ulong)row * D + col]);
            s[j / 4] += v[i][j].x + v[i][j].y;
        }
        #pragma clang loop unroll(full)
        for (ushort b = 0; b < PTN / 4; ++b) deposit(&part[0][8 * i + f.y][nb0 / 4 + b], s[b], lane);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < PM / 8; ++i) {
        float q[PTN / 4] = {};
        mean[i] = total(part[0][8 * i + f.y]) / D;
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < PTN; ++j) {
            const float2 d = v[i][j] - mean[i];
            q[j / 4] += d.x * d.x + d.y * d.y;
        }
        #pragma clang loop unroll(full)
        for (ushort b = 0; b < PTN / 4; ++b) deposit(&part[1][8 * i + f.y][nb0 / 4 + b], q[b], lane);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < PM / 8; ++i) {
        const float inv = rsqrt(total(part[1][8 * i + f.y]) / D + 1e-5f);
        const uint row = row0 + 8 * i + f.y;
        if (row >= m) continue;
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < PTN; ++j) {
            const ushort col = (nb0 + j) * 8 + f.x;
            const ulong at = (ulong)row * D + col;
            *(device half2*)&y[at] = half2(v[i][j]);
            *(device half2*)&normed[at] = half2((v[i][j] - mean[i]) * inv * float2(ln[col], ln[col + 1]) + float2(ln[D + col], ln[D + col + 1]));
        }
    }
}

#define PANEL(name, PM, PSG, PTN)                                                                                    \
    kernel void name(device const half* a [[buffer(0)]], device const half* w [[buffer(1)]],                          \
                     device const float* bias [[buffer(2)]], device half* y [[buffer(3)]],                              \
                     device half* normed [[buffer(4)]], device const float* ln [[buffer(5)]],                           \
                     constant uint3& dims [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],                     \
                     ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],          \
                     ushort lane [[thread_index_in_simdgroup]]) {                                                       \
        threadgroup half rows[PM * LDP];                                                                                \
        threadgroup float part[2][PM][D / 32];                                                                          \
        panel<PM, PSG, PTN>(a, w, bias, y, normed, ln, dims, tg, tid, sg, lane, rows, part);                            \
    }

kernel void attend(device const half* qkv [[buffer(0)]], device const float* bias [[buffer(1)]],
                   device half* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup half weight[16 * TOKENS];
    const uint pos = tg.x, head = tg.y, q0 = 16 * tg.z;
    device const half* q = qkv + (ulong)(pos * HEADS + head) * 3 * TOKENS * DH;
    device const half* key = q + TOKENS * DH;
    device const half* value = q + 2 * TOKENS * DH;
    simdgroup_float8x8 s[2][TOKENS / 8];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < 2; ++i)
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < TOKENS / 8; ++j) s[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    #pragma clang loop unroll(full)
    for (ushort k = 0; k < DH; k += 8) {
        simdgroup_half8x8 a0, a1, kt;
        simdgroup_load(a0, q + q0 * DH + k, DH);
        simdgroup_load(a1, q + (q0 + 8) * DH + k, DH);
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < TOKENS / 8; ++j) {
            simdgroup_load(kt, key + 8 * j * DH + k, DH, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s[0][j], a0, kt, s[0][j]);
            simdgroup_multiply_accumulate(s[1][j], a1, kt, s[1][j]);
        }
    }
    const ushort2 f = owned(lane);
    const float scale = rsqrt(float(DH));
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < 2; ++i) {
        const ushort r = 8 * i + f.y;
        device const float* bt = bias + (head * TOKENS + q0 + r) * TOKENS + f.x;
        float2 e[TOKENS / 8];
        float top = -INFINITY, sum = 0.0f;
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < TOKENS / 8; ++j) {
            const thread auto& x = s[i][j].thread_elements();
            e[j] = float2(x[0], x[1]) * scale + float2(bt[8 * j], bt[8 * j + 1]);
            top = max(top, max(e[j].x, e[j].y));
        }
        top = max(top, simd_shuffle_xor(top, 1));
        top = max(top, simd_shuffle_xor(top, 8));
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < TOKENS / 8; ++j) {
            e[j] = exp(e[j] - top);
            sum += e[j].x + e[j].y;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 8);
        const float inv = 1.0f / sum;
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < TOKENS / 8; ++j) *(threadgroup half2*)&weight[r * TOKENS + 8 * j + f.x] = half2(e[j] * inv);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 o[2][DH / 8];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < 2; ++i)
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < DH / 8; ++j) o[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    #pragma clang loop unroll(full)
    for (ushort kb = 0; kb < TOKENS / 8; ++kb) {
        simdgroup_half8x8 p0, p1, v;
        simdgroup_load(p0, weight + 8 * kb, TOKENS);
        simdgroup_load(p1, weight + 8 * TOKENS + 8 * kb, TOKENS);
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < DH / 8; ++j) {
            simdgroup_load(v, value + 8 * kb * DH + 8 * j, DH);
            simdgroup_multiply_accumulate(o[0][j], p0, v, o[0][j]);
            simdgroup_multiply_accumulate(o[1][j], p1, v, o[1][j]);
        }
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < 2; ++i)
        #pragma clang loop unroll(full)
        for (ushort j = 0; j < DH / 8; ++j) {
            const thread auto& e = o[i][j].thread_elements();
            *(device half2*)&out[((ulong)(q0 + 8 * i + f.y) * n + pos) * D + head * DH + 8 * j + f.x] = half2(float2(e[0], e[1]));
        }
}

kernel void heads(device const half* hexes [[buffer(0)]], device const half* coins [[buffer(1)]],
                  device const half* state [[buffer(2)]], device const float* kind [[buffer(3)]],
                  device const float* none [[buffer(4)]], device const float* value_w [[buffer(5)]],
                  device const float* value_b [[buffer(6)]], device const float* move_w [[buffer(7)]],
                  device const uchar* desc [[buffer(8)]], device const uint* off [[buffer(9)]],
                  device float* out [[buffer(10)]], constant uint& n [[buffer(11)]],
                  uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                  ushort lane [[thread_index_in_simdgroup]]) {
    const uint first = tg * HEAD_ROWS, rows = min(n - first, (uint)HEAD_ROWS);
    for (uint r = sg; r < rows; r += 8) {
        device const half* v = state + (ulong)(first + r) * 2 * D + D;
        float acc = 0.0f;
        for (ushort c = lane; c < D; c += 32) acc += gelu(v[c]) * value_w[c];
        acc = simd_sum(acc);
        if (lane == 0) out[first + r] = 1.0f - 2.0f / (exp(acc + value_b[0]) + 1.0f);
    }
    device float* prior = out + n;
    for (uint i = off[first] + sg; i < off[first + rows]; i += 8) {
        uint row = first;
        while (off[row + 1] <= i) ++row;
        device const uchar* d = desc + (ulong)i * ACT_BYTES;
        device const half* pick[ACT_BYTES - 1];
        for (ushort s = 0; s < ACT_BYTES - 1; ++s) {
            const bool coin = s < 2;
            const uchar at = d[1 + s];
            pick[s] = at < (coin ? NTYPE : N_HEXES)
                          ? (coin ? coins + ((ulong)at * n + row) * 2 * D + s * D : hexes + ((ulong)at * n + row) * 3 * D + (s - 2) * D)
                          : nullptr;
        }
        device const half* st = state + (ulong)row * 2 * D;
        float acc = 0.0f;
        for (ushort c = lane; c < D; c += 32) {
            float v = kind[d[0] * D + c] + st[c];
            for (ushort s = 0; s < ACT_BYTES - 1; ++s) v += pick[s] ? float(pick[s][c]) : none[s * D + c];
            acc += gelu(v) * move_w[c];
        }
        acc = simd_sum(acc);
        if (lane == 0) prior[i] = acc;
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint r = sg; r < rows; r += 8) {
        const uint lo = off[first + r], hi = off[first + r + 1];
        float top = -INFINITY, mine = 0.0f;
        for (uint i = lo + lane; i < hi; i += 32) top = max(top, prior[i]);
        top = simd_max(top);
        for (uint i = lo + lane; i < hi; i += 32) mine += prior[i] = exp(prior[i] - top);
        const float scale = 1.0f / simd_sum(mine);
        for (uint i = lo + lane; i < hi; i += 32) prior[i] *= scale;
    }
}
