use super::*;

#[derive(Default)]
pub(super) struct Batch {
    pub trees: Wire<u64>,
    pub part: Wire<i32>,
    pub local: Wire<i32>,
    pub prime: Wire<u32>,
    queries: Wire<u32>,
    touched: Wire<i32>,
    pub rows: usize,
    pub parts: u32,
    levels: usize,
}

impl Batch {
    pub(super) fn carve(plan: &mut Plan, n: usize, slots: usize, cfg: &Cfg) -> Res<Batch> {
        let rows = cfg.budget.cap(Ent::Row);
        Ok(Batch {
            trees: plan.wire(slots * DESC)?,
            part: plan.wire(slots * rows)?,
            local: plan.wire(slots * rows)?,
            prime: plan.wire(12 * TILE + n * cfg.budget.cap(Ent::Cell))?,
            queries: plan.wire(4 * slots * cfg.s.max(1) as usize)?,
            touched: plan.wire(slots)?,
            ..Batch::default()
        })
    }
}

const CFR_THREADS: u32 = 1024;

impl Card {
    pub(super) fn lay(&self, solves: &[usize]) -> Res<()> {
        let mut batch = self.batch.lock();
        let mut desc: Vec<u64> = Vec::with_capacity(solves.len() * DESC);
        let (mut part_of_row, mut local_row): (Vec<i32>, Vec<i32>) = (Vec::new(), Vec::new());
        let mut levels = 0;
        for (part, &solve) in solves.iter().enumerate() {
            let b = self
                .solves
                .get(solve)
                .map(|m| m.lock())
                .filter(|b| b.host_coff.len() == 2 * b.rows + 1)
                .ok_or_else(|| format!("solve {solve} has no resident tree"))?;
            desc.extend_from_slice(&b.describe(&self.stream));
            part_of_row.extend(std::iter::repeat_n(part as i32, b.rows));
            local_row.extend(0..b.rows as i32);
            levels = levels.max(b.level_start.len().saturating_sub(1));
        }
        let s = &self.stream;
        batch.trees.put(s, desc.len(), copy(&desc))?;
        batch.part.put(s, part_of_row.len(), copy(&part_of_row))?;
        batch.local.put(s, local_row.len(), copy(&local_row))?;
        batch.rows = part_of_row.len();
        batch.parts = solves.len() as u32;
        batch.levels = levels;
        Ok(())
    }

    fn reach(&self, b: &Batch, avg: i32) -> Res<()> {
        unsafe {
            self.stream
                .launch_builder(&self.k.reach)
                .arg(b.trees.buf()).arg(&avg)
                .launch_unit(LaunchConfig {
                    grid_dim: (b.parts, 1, 1),
                    block_dim: (CFR_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(err)
    }

    fn cfr(&self, b: &Batch, sims: usize, puct: f32, avg: i32, k: Cfr) -> Res<()> {
        let draws = sims * crate::search::TRIES;
        let mut sc = self.scratch.lock();
        let Scratch { leaves, queries, .. } = &mut *sc;
        carved!(leaves, queries);
        let (parts_i, sims_i, tries_i, depth_i) =
            (b.parts as i32, sims as i32, crate::search::TRIES as i32, b.levels as i32);
        unsafe {
            self.stream
                .launch_builder(&self.k.cfr)
                .arg(b.trees.buf()).arg(leaves).arg(b.queries.buf()).arg(queries)
                .arg(&parts_i).arg(&sims_i).arg(&puct).arg(&tries_i).arg(&depth_i).arg(&avg)
                .arg(&k.alpha).arg(&k.beta).arg(&k.gamma).arg(&k.predict)
                .launch_unit(LaunchConfig {
                    grid_dim: (b.parts, 1, 1),
                    block_dim: (CFR_THREADS, 1, 1),
                    shared_mem_bytes: (4 * draws * (b.levels + 2)) as u32,
                })
        }
        .map_err(err)
    }

    pub(super) fn iterate(&self, calls: &[Call], mine: &[usize], out: &mut Vec<(usize, Reply)>) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        let (puct, k) = match &calls[mine[0]] {
            Call::Iterate { puct, cfr, .. } => (*puct, *cfr),
            _ => unreachable!("iterate shard holds only iterate calls"),
        };
        let mut sims = 0usize;
        let mut rounds = 0usize;
        let mut query_at = vec![0usize; calls.len()];
        let mut query_len = vec![0usize; calls.len()];
        let mut flat: Vec<u32> = Vec::new();
        let mut query_total = 0usize;
        for &i in mine {
            let Call::Iterate { solve, step, iters, expand, query, cfr, puct: p } = &calls[i] else {
                unreachable!("iterate shard holds only iterate calls")
            };
            assert_eq!((*cfr, *p), (k, puct), "a round mixes two regret rules");
            let mut b = self.slot(*solve);
            b.step = *step;
            b.todo = *iters;
            b.nexpand = *expand;
            b.query_at = flat.len() / 4;
            b.nquery = query.len();
            sims = sims.max(*expand);
            rounds = rounds.max(*iters);
            query_at[i] = query_total;
            for q in query {
                debug_assert!((q.iter as usize) < *iters, "query pick is outside its solve's live iterations");
                flat.extend([q.iter, q.reach, q.len, query_total as u32]);
                query_total += q.len as usize;
            }
            query_len[i] = query_total - query_at[i];
        }
        let solves: Vec<usize> = mine.iter().map(|&i| calls[i].solve()).collect();
        self.lay(&solves)?;
        let mut b = self.batch.lock();
        b.queries.put(&self.stream, flat.len(), copy(&flat))?;
        self.scratch.lock().queries.room(query_total)?;
        self.scratch.lock().leaves.room((rounds * b.parts as usize * sims).max(1))?;
        self.reach(&b, 0)?;
        self.opinions(&b)?;
        self.cfr(&b, sims, puct, 0, k)?;
        let each = b.parts as usize * sims;
        let host = self.sampled(rounds * each)?;
        let query_host = if query_total == 0 {
            Vec::new()
        } else {
            let scratch = self.scratch.lock();
            let src = scratch.queries.buf.as_ref().expect("query scratch is carved");
            self.down_f.lock().recv(&self.stream, &src.slice(..query_total))?
        };
        drop(b);
        for (part, &i) in mine.iter().enumerate() {
            let Call::Iterate { iters, expand, .. } = &calls[i] else { unreachable!() };
            let mut leaves = Vec::with_capacity(iters * expand);
            for phase in 0..*iters {
                let at = phase * each + part * sims;
                leaves.extend_from_slice(&host[at..at + expand]);
            }
            let q = query_at[i];
            let c = query_host[q..q + query_len[i]].to_vec();
            out.push((i, Reply { c, leaves, ..Default::default() }));
        }
        Ok(())
    }

    pub(super) fn read(&self, calls: &[Call], mine: &[usize], out: &mut Vec<(usize, Reply)>) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        let solves: Vec<usize> = mine.iter().map(|&i| calls[i].solve()).collect();
        self.lay(&solves)?;
        let touched: Vec<i32> = mine
            .iter()
            .map(|&i| match &calls[i] {
                Call::Read { touched, .. } => (touched[0] as i32) | ((touched[1] as i32) << 1),
                _ => unreachable!("read shard holds only read calls"),
            })
            .collect();
        {
            let mut b = self.batch.lock();
            b.touched.put(&self.stream, touched.len(), copy(&touched))?;
            unsafe {
                self.stream
                    .launch_builder(&self.k.finish)
                    .arg(b.trees.buf()).arg(b.touched.buf())
                    .launch_unit(LaunchConfig {
                        grid_dim: (b.parts, 1, 1),
                        block_dim: (CFR_THREADS, 1, 1),
                        shared_mem_bytes: 0,
                    })
            }
            .map_err(err)?;
        }
        let want: Vec<usize> = mine
            .iter()
            .filter(|&&i| matches!(&calls[i],
                Call::Read { vals_at, .. } if vals_at[0].1 > 0 || vals_at[1].1 > 0))
            .map(|&i| calls[i].solve())
            .collect();
        if !want.is_empty() {
            for &solve in &want {
                let mut b = self.slot(solve);
                b.todo = 1;
                b.nexpand = 0;
                b.nquery = 0;
            }
            self.lay(&want)?;
            let b = self.batch.lock();
            self.reach(&b, 1)?;
            self.opinions(&b)?;
            self.cfr(&b, 0, 0.0, 1, Cfr::LINEAR)?;
        }
        let mut h = self.down_f.lock();
        for &i in mine {
            let Call::Read { solve, vals_at, policy_at, .. } = &calls[i] else {
                unreachable!("read shard holds only read calls")
            };
            let s = self.slot(*solve);
            let mut root = Vec::new();
            for &(at, n) in vals_at {
                root.extend(s.ent[Ent::Reach as usize].get_f32(&self.stream, R_VALS, at as usize, n as usize, &mut h)?);
            }
            let cells = &s.ent[Ent::Cell as usize];
            let (at, n) = (policy_at.0 as usize, policy_at.1 as usize);
            let policy = cells.get_f32(&self.stream, C_SUM, at, n, &mut h)?;
            let q = cells.get_f32(&self.stream, C_QVAL, at, n, &mut h)?;
            out.push((i, Reply { a: root, b: policy, q, ..Default::default() }));
        }
        Ok(())
    }

    fn opinions(&self, b: &Batch) -> Res<()> {
        let (trees, part, local) = (b.trees.buf(), b.part.buf(), b.local.buf());
        let stride = b.rows;
        let l = &self.layout;
        let belief_span = Span {
            w: l.value_hidden.w + (ATTN + D) * ATTN,
            b: usize::MAX,
            i: 2 * ATTN,
            o: ATTN,
        };
        let out_b = self.b.slice(l.value_out.b..l.value_out.b + BINS);
        for q0 in (0..stride).step_by(TILE) {
            let n = TILE.min(stride - q0);
            let q0_i = q0 as i32;
            let rows = 2 * n;
            let rows_i = rows as i32;
            let mut scratch = self.scratch.lock();
            scratch.attn_mix.room(rows * 2 * ATTN)?;
            scratch.h.room(rows * ATTN)?;
            let Scratch { attn_mix, h, .. } = &mut *scratch;
            carved!(attn_mix, h);
            unsafe {
                self.stream.launch_builder(&self.k.belief)
                    .arg(trees).arg(part).arg(local).arg(&mut *attn_mix)
                    .arg(&rows_i).arg(&q0_i)
                    .launch_unit(warp_rows(rows))
            }.map_err(err)?;
            self.hlin32(belief_span, &*attn_mix, rows, h)?;
            unsafe {
                self.stream.launch_builder(&self.k.value)
                    .arg(trees).arg(part).arg(local).arg(&*h)
                    .arg(&self.value_w).arg(&out_b)
                    .arg(&rows_i).arg(&q0_i)
                    .launch_unit(LaunchConfig {
                        grid_dim: (rows.div_ceil(VALUE_WARPS) as u32, 1, 1),
                        block_dim: (32, VALUE_WARPS as u32, 1),
                        shared_mem_bytes: 0,
                    })
            }.map_err(err)?;
        }
        Ok(())
    }

    fn sampled(&self, n: usize) -> Res<Vec<u32>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut sc = self.scratch.lock();
        let out = sc.leaves.room(n)?;
        self.down.lock().recv(&self.stream, &out.slice(0..n))
    }
}
