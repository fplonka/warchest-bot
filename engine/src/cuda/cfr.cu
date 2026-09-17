
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define KIND_DECISION 0u
#define CFR_THREADS 1024
#define SCAN_CAP 8192
#define SCAN_PER (SCAN_CAP / CFR_THREADS)
static_assert(SCAN_CAP % CFR_THREADS == 0, "the level scan gives every thread the same run");

struct CfrShared {
    unsigned prefix[SCAN_CAP + 1];
    unsigned warp_sum[CFR_THREADS / 32];
};

__device__ __forceinline__ unsigned sweep_count(const Tree& t, int sweep, unsigned node) {
    const int p = sweep & 1;
    const unsigned n = t.nc[2 * node + p], kind = t.kind[node];
    const bool own = (int)t.player[node] == p;
    if (sweep < 2) return n;
    if (sweep < 4) return kind == KIND_CHANCE || !own ? n : 0u;
    return kind == KIND_DECISION && own ? n : 0u;
}

extern "C" __global__ void __launch_bounds__(CFR_THREADS)
k_level_prefix(const Tree* trees) {
    __shared__ unsigned warp_sum[CFR_THREADS / 32], total;
    const Tree& t = trees[blockIdx.z];
    const int level = blockIdx.x, sweep = blockIdx.y;
    if (level >= (int)t.nlevels) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const unsigned begin = t.level_start[level], end = t.level_start[level + 1];
    unsigned carry = 0;
    for (unsigned chunk = begin; chunk < end; chunk += CFR_THREADS) {
        const unsigned k = chunk + tid;
        const unsigned mine = k < end ? sweep_count(t, sweep, t.level_node[k]) : 0u;
        unsigned inclusive = mine;
        for (int o = 1; o < 32; o <<= 1) {
            unsigned v = __shfl_up_sync(0xffffffff, inclusive, o);
            if (lane >= o) inclusive += v;
        }
        if (lane == 31) warp_sum[warp] = inclusive;
        __syncthreads();
        if (warp == 0) {
            unsigned v = warp_sum[lane], acc = v;
            for (int o = 1; o < 32; o <<= 1) {
                unsigned w = __shfl_up_sync(0xffffffff, acc, o);
                if (lane >= o) acc += w;
            }
            warp_sum[lane] = acc - v;
            if (lane == 31) total = acc;
        }
        __syncthreads();
        if (k < end) t.prefix[6 * k + sweep] = carry + warp_sum[warp] + inclusive - mine;
        carry += total;
        __syncthreads();
    }
}

template <int LANES, int U, typename Body>
__device__ void level_items(CfrShared& s, const Tree& t, int sweep, unsigned begin, unsigned end,
                            Body body) {
    const int tid = threadIdx.x;
    const int sub = tid % LANES, group = tid / LANES, groups = CFR_THREADS / LANES;
    const unsigned* prefix = t.prefix + sweep;
    for (unsigned chunk = begin; chunk < end; chunk += SCAN_CAP) {
        const unsigned n = min((unsigned)SCAN_CAP, end - chunk);
        const unsigned first = prefix[6 * chunk];
        for (unsigned k = tid; k < n; k += CFR_THREADS) s.prefix[k] = prefix[6 * (chunk + k)] - first;
        if (tid == 0) s.prefix[n] = prefix[6 * (chunk + n - 1)] - first + sweep_count(t, sweep, t.level_node[chunk + n - 1]);
        __syncthreads();
        const unsigned total = s.prefix[n];
        for (unsigned base = 0; base < total; base += U * groups) {
            unsigned node[U], c[U];
            bool live[U];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                unsigned idx = base + u * groups + group;
                live[u] = idx < total;
                unsigned lo = 0, hi = n;
                while (lo + 1 < hi) {
                    unsigned mid = (lo + hi) >> 1;
                    if (s.prefix[mid] <= min(idx, total - 1)) lo = mid; else hi = mid;
                }
                node[u] = t.level_node[chunk + lo];
                c[u] = idx - s.prefix[lo];
            }
            body(node, c, sub, live);
        }
        __syncthreads();
    }
}

__device__ __forceinline__ float sum8(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

#define CFR_U 4

__device__ void reach_level(CfrShared& s, const Tree& t, int level, int p, const float* strat) {
    level_items<1, CFR_U>(s, t, p, t.level_start[level], t.level_start[level + 1],
        [&](const unsigned (&node)[CFR_U], const unsigned (&e)[CFR_U], int, const bool (&live)[CFR_U]) {
            float v[CFR_U];
            unsigned dst[CFR_U];
#pragma unroll
            for (int u = 0; u < CFR_U; ++u) {
                v[u] = 0.0f;
                dst[u] = 0u;
                if (!live[u]) continue;
                unsigned c = e[u];
                unsigned par = t.parent[node[u]];
                unsigned src = rbase(t, par, p);
                dst[u] = rbase(t, node[u], p) + c;
                if ((int)t.player[par] != p) {
                    v[u] = t.reach[src + c];
                    continue;
                }
                unsigned rb = t.rev_base[node[u]];
                if (rb != NO_ROW) {
                    unsigned a = t.rev_start[rb + c], b = t.rev_start[rb + c + 1];
                    for (unsigned k = a; k < b; ++k) v[u] += t.reach[src + t.rev_src[k]] * strat[t.rev_cell[k]];
                } else {
                    unsigned db = t.rvd_base[node[u]];
                    if (db != NO_ROW) {
                        unsigned a = t.rvd_start[db + c], b = t.rvd_start[db + c + 1];
                        for (unsigned k = a; k < b; ++k) v[u] += t.reach[src + t.rvd_src[k]] * t.rvd_p[k];
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < CFR_U; ++u)
                if (live[u]) t.reach[dst[u]] = v[u];
        });
}

__device__ void reach_pass(CfrShared& s, const Tree& t, int p, int avg) {
    const float* strat = avg ? t.avg : t.cur;
    for (int level = 1; level < (int)t.nlevels; ++level) reach_level(s, t, level, p, strat);
}

__device__ void leaf_values(const Tree& t, unsigned traverser) {
    const int sub = threadIdx.x % 8, group = threadIdx.x / 8, groups = CFR_THREADS / 8;
    const unsigned rows = (unsigned)t.nrows, total = rows + (unsigned)t.nterm, opp = 1 - traverser;
    for (unsigned base = 0; base < total; base += CFR_U * groups) {
        unsigned node[CFR_U];
        float mass[CFR_U];
#pragma unroll
        for (int u = 0; u < CFR_U; ++u) {
            unsigned item = base + u * groups + group;
            bool live = item < total;
            node[u] = live ? (item < rows ? t.leaf_node[item] : t.term[item - rows]) : 0u;
            unsigned n = live ? t.nc[2 * node[u] + opp] : 0u, ra = live ? rbase(t, node[u], opp) : 0u;
            mass[u] = 0.0f;
            for (unsigned c = sub; c < n; c += 8) mass[u] += t.reach[ra + c];
        }
#pragma unroll
        for (int u = 0; u < CFR_U; ++u) {
            unsigned item = base + u * groups + group;
            mass[u] = sum8(mass[u]);
            if (item >= total) continue;
            unsigned m = t.nc[2 * node[u] + traverser], vo = t.voff[node[u]];
            float* vals = t.vals + traverser * t.nvals;
            if (item < rows) {
                const float* opinion = t.opinion + traverser * t.nvals;
                for (unsigned c = sub; c < m; c += 8) vals[vo + c] = opinion[vo + c] * mass[u];
            } else {
                float sign = t.player[node[u]] == traverser ? t.util[node[u]] : -t.util[node[u]];
                for (unsigned c = sub; c < m; c += 8) vals[vo + c] = sign * mass[u];
            }
        }
    }
}

__device__ void backprop_level(CfrShared& s, const Tree& t, int level, unsigned traverser, int avg,
                               int iter, float alpha, float beta, float gamma, float predict) {
    float m = (float)(t.step + (unsigned long long)iter) + 1.0f;
    float da = cfr_factor(m, alpha), db = cfr_factor(m, beta);
    float dg = powf((m - 1.0f) / m, gamma);
    const unsigned begin = t.level_start[level], split = t.level_split[level];
    float* vals = t.vals + traverser * t.nvals;
    level_items<1, CFR_U>(s, t, 2 + traverser, begin, split,
        [&](const unsigned (&node)[CFR_U], const unsigned (&e)[CFR_U], int, const bool (&live)[CFR_U]) {
            float v[CFR_U];
            float* dst[CFR_U];
#pragma unroll
            for (int u = 0; u < CFR_U; ++u) {
                v[u] = 0.0f;
                dst[u] = nullptr;
                if (!live[u]) continue;
                unsigned c = e[u];
                unsigned me = t.player[node[u]], vi = t.voff[node[u]];
                bool chance = t.kind[node[u]] == KIND_CHANCE;
                dst[u] = vals + vi + c;
                if (chance) {
                    unsigned cv = t.voff[t.child[t.child_at[node[u]]]];
                    if (me != traverser) {
                        v[u] = vals[cv + c];
                        continue;
                    }
                    unsigned base = t.draw_base[node[u]];
                    unsigned a = t.draw_start[base + c], b = t.draw_start[base + c + 1];
                    for (unsigned k = a; k < b; ++k) v[u] += t.draw_p[k] * vals[cv + t.draw_to[k]];
                    continue;
                }
                unsigned a = t.child_at[node[u]], k = t.child_n[node[u]];
                for (unsigned j = a; j < a + k; ++j) v[u] += vals[t.voff[t.child[j]] + c];
            }
#pragma unroll
            for (int u = 0; u < CFR_U; ++u)
                if (live[u]) *dst[u] = v[u];
        });
    const float* strat = avg ? t.avg : t.cur;
    level_items<8, 1>(s, t, 4 + traverser, begin, split,
        [&](const unsigned (&nodes)[1], const unsigned (&cs)[1], int sub, const bool (&lives)[1]) {
            const unsigned node = nodes[0], c = cs[0];
            const bool live = lives[0];
            const unsigned me = traverser;
            unsigned so = live ? t.soff[node] : 0u, lb = live ? t.legal_base[node] : 0u;
            unsigned a = live ? t.legal_off[lb + c] : 0u, b = live ? t.legal_off[lb + c + 1] : 0u;
            float base = 0.0f;
            for (unsigned cell = a + sub; cell < b; cell += 8) {
                unsigned vc = t.cell_val[so + cell];
                float av = vc == NO_ROW ? 0.0f : vals[vc];
                t.qval[so + cell] = av;
                base += av * strat[so + cell];
            }
            base = sum8(base);
            if (live && sub == 0) vals[t.voff[node] + c] = base;
            if (avg) return;
            float reach = live ? t.reach[rbase(t, node, me) + c] : 0.0f;
            float total = 0.0f;
            for (unsigned cell = a + sub; cell < b; cell += 8) {
                float delta = t.qval[so + cell] - base;
                float old = t.regret[so + cell];
                float r = old * (old > 0.0f ? da : db) + delta;
                t.regret[so + cell] = r;
                t.sum[so + cell] = t.sum[so + cell] * dg + reach * t.cur[so + cell];
                float v = fmaxf(r + predict * delta, 0.0f);
                t.cur[so + cell] = v;
                total += v;
            }
            total = sum8(total);
            float scale = total > SMOOTH ? 1.0f / total : 0.0f;
            float flat = 1.0f / (float)max(b - a, 1u);
            for (unsigned cell = a + sub; cell < b; cell += 8)
                t.cur[so + cell] = total > SMOOTH ? t.cur[so + cell] * scale : flat;
        });
}

__device__ __forceinline__ unsigned long long rng_next(unsigned long long* s) {
    unsigned long long x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

__device__ __forceinline__ double rng_unit(unsigned long long* s) {
    return (double)(rng_next(s) >> 11) * (1.0 / 9007199254740992.0);
}

__device__ __forceinline__ float pick_sum(const float* w, int n) {
    float total = 0.0f;
    for (int i = threadIdx.x % 32; i < n; i += 32) total += fmaxf(w[i], 0.0f);
    return warp_sum(total);
}

__device__ int pick_from(const float* w, int n, float total, unsigned long long* s) {
    if (!(total > 0.0f)) return n > 0 ? (int)(rng_next(s) % (unsigned long long)n) : 0;
    double needle = rng_unit(s) * (double)total;
    for (int i = 0; i < n; ++i) {
        needle -= (double)fmaxf(w[i], 0.0f);
        if (needle < 0.0) return i;
    }
    return n - 1;
}

__device__ __forceinline__ int pick(const float* w, int n, unsigned long long* s) {
    return pick_from(w, n, pick_sum(w, n), s);
}

__device__ bool live_cell(const Tree& t, unsigned int so, unsigned int cell) {
    return t.legal_trans[so + cell] != NO_TRANS
        && t.exhausted[t.legal_child[so + cell]] == 0u;
}

__device__ unsigned int pick_live(const Tree& t, unsigned int so, unsigned int a,
                                  unsigned int b, const float* w,
                                  unsigned long long* s) {
    const int lane = threadIdx.x % 32;
    float total = 0.0f, n = 0.0f;
    for (unsigned int i = a + lane; i < b; i += 32) {
        if (live_cell(t, so, i)) { total += fmaxf(w[i - a], 0.0f); n += 1.0f; }
    }
    total = warp_sum(total);
    int live = (int)warp_sum(n);
    if (live == 0) return NO_ROW;
    if (!(total > 0.0f)) {
        int k = (int)(rng_next(s) % (unsigned long long)live);
        for (unsigned int i = a; i < b; ++i) {
            if (!live_cell(t, so, i)) continue;
            if (k == 0) return i;
            --k;
        }
        return NO_ROW;
    }
    double needle = rng_unit(s) * (double)total;
    unsigned int last = NO_ROW;
    for (unsigned int i = a; i < b; ++i) {
        if (!live_cell(t, so, i)) continue;
        last = i;
        needle -= (double)fmaxf(w[i - a], 0.0f);
        if (needle < 0.0) return i;
    }
    return last;
}

__device__ unsigned int puct_choice(const Tree& t, unsigned int node, unsigned int a,
                                    unsigned int b, int opp, float c_puct) {
    const int lane = threadIdx.x % 32;
    unsigned int so = t.soff[node], ra = rbase(t, node, opp);
    unsigned int nc = t.nc[2 * node + opp];
    float mass = 0.0f;
    for (unsigned int i = lane; i < nc; i += 32) mass += t.reach[ra + i];
    mass = warp_sum(mass);
    float scale = mass > SMOOTH ? __fdiv_rn(1.0f, mass) : 0.0f;
    float total = 0.0f;
    for (unsigned int cell = a + lane; cell < b; cell += 32)
        total += t.visits[so + cell];
    total = warp_sum(total);
    float explore = __fmul_rn(c_puct, sqrtf(fmaxf(total, 0.0f)));
    unsigned int best = NO_ROW;
    float score = neg_inf();
    for (unsigned int cell = a + lane; cell < b; cell += 32) {
        if (!live_cell(t, so, cell)) continue;
        float u = __fdiv_rn(__fmul_rn(explore, t.prior[so + cell]),
                            __fadd_rn(1.0f, t.visits[so + cell]));
        float v = __fadd_rn(__fmul_rn(t.qval[so + cell], scale), u);
        if (v > score) { score = v; best = cell; }
    }
    for (int k = 16; k > 0; k >>= 1) {
        float os = __shfl_xor_sync(0xffffffff, score, k);
        unsigned int oc = __shfl_xor_sync(0xffffffff, best, k);
        if (os > score || (os == score && oc < best)) { score = os; best = oc; }
    }
    return best;
}

__device__ __forceinline__ unsigned long long rng_seed(unsigned long long seed,
                                                       unsigned long long iter,
                                                       unsigned long long draw) {
    unsigned long long z = seed + iter * 0x9E3779B97F4A7C15ULL + (draw + 1) * 0xD1B54A32D192ED03ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return z ? z : 1;
}

__device__ void expand(const Tree& t, unsigned int* path, unsigned int* out, int part,
                       int sims, float c_puct, int iter, int each, int tries, int depth) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = CFR_THREADS / 32;
    const int draws = sims * tries;
    unsigned int* cand = path + (size_t)draws * depth;
    unsigned int* len = cand + draws;
    unsigned int* taken = out + (size_t)iter * each + (size_t)part * sims;
    for (int i = threadIdx.x; i < sims; i += CFR_THREADS) taken[i] = NO_ROW;
    __syncthreads();
    int want = (int)t.nexpand;
    unsigned int n0 = t.nc[0], n1 = t.nc[1];
    const float* root = t.reach + t.roff[0];
    for (int draw = warp; draw < draws; draw += warps) {
        unsigned int* mine = path + (size_t)draw * depth;
        unsigned int found = NO_ROW;
        int steps = 0;
        if (draw < want * tries) {
            unsigned long long s = rng_seed(*t.seed, t.step + iter, draw);
            int c[2];
            c[0] = pick(root, (int)n0, &s);
            c[1] = pick(root + n0, (int)n1, &s);
            unsigned int node = 0;
            for (;;) {
                unsigned int k = t.kind[node];
                if (k == KIND_LEAF) {
                    found = node;
                    break;
                }
                unsigned int me = t.player[node];
                if (k == KIND_CHANCE) {
                    unsigned int base = t.draw_base[node];
                    unsigned int a = t.draw_start[base + c[me]];
                    unsigned int b = t.draw_start[base + c[me] + 1];
                    int j = pick(t.draw_p + a, (int)(b - a), &s);
                    c[me] = (int)t.draw_to[a + j];
                    node = t.child[t.child_at[node]];
                    continue;
                }
                unsigned int lb = t.legal_base[node], so = t.soff[node];
                unsigned int a = t.legal_off[lb + c[me]], b = t.legal_off[lb + c[me] + 1];
                unsigned int cell;
                if (rng_unit(&s) < 0.5) {
                    cell = puct_choice(t, node, a, b, 1 - me, c_puct);
                } else {
                    bool any_sum = false;
                    for (unsigned int q = a + lane; q < b; q += 32) any_sum |= t.sum[so + q] > 0.0f;
                    any_sum = __any_sync(0xffffffff, any_sum);
                    const float* row = any_sum ? t.sum + so + a : t.cur + so + a;
                    cell = pick_live(t, so, a, b, row, &s);
                }
                if (cell == NO_ROW) break;
                if (lane == 0) mine[steps] = so + cell;
                ++steps;
                c[me] = (int)t.legal_trans[so + cell];
                node = t.legal_child[so + cell];
            }
        }
        if (lane == 0) {
            cand[draw] = found;
            len[draw] = steps;
        }
    }
    __syncthreads();
    for (int draw = warp; draw < draws; draw += warps)
        for (int d = lane; d < (int)len[draw]; d += 32)
            atomicAdd(&t.visits[path[(size_t)draw * depth + d]], 1.0f);
    if (warp != 0) return;
    int got = 0;
    for (int draw = 0; draw < draws && got < want; ++draw) {
        unsigned int found = cand[draw];
        if (found == NO_ROW) continue;
        bool dup = false;
        for (int k = lane; k < (iter + 1) * sims; k += 32) {
            const unsigned int* r = out + (size_t)(k / sims) * each + (size_t)part * sims;
            dup |= r[k % sims] == found;
        }
        if (__any_sync(0xffffffff, dup)) continue;
        if (lane == 0) taken[got] = found;
        __syncwarp();
        ++got;
    }
}

extern "C" __global__ void __launch_bounds__(CFR_THREADS)
k_reach(const Tree* trees, int avg) {
    __shared__ CfrShared s;
    reach_pass(s, trees[blockIdx.x], blockIdx.y, avg);
}

extern "C" __global__ void __launch_bounds__(CFR_THREADS)
k_cfr(const Tree* trees, unsigned int* out, const unsigned int* queries, float* query_out,
      int parts, int sims, float c_puct, int tries, int depth, int rounds, int avg,
      float alpha, float beta, float gamma, float predict) {
    __shared__ CfrShared s;
    extern __shared__ unsigned int path[];
    const int part = blockIdx.x, tr = blockIdx.y;
    const Tree& t = trees[part];
    const int each = parts * sims;
    cg::grid_group grid = cg::this_grid();
    for (int iter = 0; iter < rounds; ++iter) {
        const bool live = iter < (int)t.todo;
        if (live && tr == 0)
            for (unsigned q = 0; q < (unsigned)t.nquery; ++q) {
                const unsigned int* pick = queries + 4 * (t.query_at + q);
                if (pick[0] != (unsigned)iter) continue;
                for (unsigned i = threadIdx.x; i < pick[2]; i += CFR_THREADS)
                    query_out[pick[3] + i] = t.reach[pick[1] + i];
            }
        if (live) {
            leaf_values(t, tr);
            __syncthreads();
            for (int level = (int)t.nlevels - 1; level >= 0; --level)
                backprop_level(s, t, level, tr, avg, iter, alpha, beta, gamma, predict);
        }
        grid.sync();
        if (sims > 0) {
            if (live && tr == 0) expand(t, path, out, part, sims, c_puct, iter, each, tries, depth);
            grid.sync();
        }
        if (iter + 1 < rounds) {
            if (iter + 1 < (int)t.todo) reach_pass(s, t, tr, avg);
            grid.sync();
        }
    }
}

extern "C" __global__ void __launch_bounds__(CFR_THREADS)
k_finish(const Tree* trees, const int* touched) {
    __shared__ CfrShared s;
    const Tree& t = trees[blockIdx.x];
    int mask = touched[blockIdx.x];
    if (mask < 0) return;
    for (int level = 0; level < (int)t.nlevels; ++level)
        for (unsigned me = 0; me < 2; ++me)
            level_items<1, 1>(s, t, 4 + me, t.level_start[level], t.level_start[level + 1],
                [&](const unsigned (&nodes)[1], const unsigned (&cs)[1], int, const bool (&lives)[1]) {
                    if (!lives[0]) return;
                    const unsigned node = nodes[0], c = cs[0];
                    unsigned so = t.soff[node], lb = t.legal_base[node];
                    unsigned a = t.legal_off[lb + c], b = t.legal_off[lb + c + 1];
                    if (!((mask >> me) & 1)) {
                        for (unsigned cell = a; cell < b; ++cell) t.avg[so + cell] = t.cur[so + cell];
                        return;
                    }
                    float sum = 0.0f;
                    for (unsigned cell = a; cell < b; ++cell) sum += t.sum[so + cell];
                    float k = (float)max(b - a, 1u);
                    for (unsigned cell = a; cell < b; ++cell)
                        t.avg[so + cell] = sum > SMOOTH ? t.sum[so + cell] / sum : 1.0f / k;
                });
}
