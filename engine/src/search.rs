use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::ops::Range;

use serde::Deserialize;

use warchest::actions::Action;
use warchest::board::{NONE, N_HEXES};
use warchest::actions::{action_coin, obs_key};
use crate::pbs::{mirror_row, pack_row, Belief, Ctx, NSLOT, NTYPE, ROW_BYTES, ROW_TO_ACT};
use warchest::rng::Rng;
use warchest::rules::resolve_chance;
use warchest::state::{mirror_hex, State, Z_FACEDOWN, Z_HAND, Z_INFLIGHT};

pub const ACT_BYTES: usize = 6;
pub const COIN_PICKS: Range<usize> = 1..3;
pub const HEX_PICKS: Range<usize> = 3..ACT_BYTES;
const C_VISIT: f32 = 50.0;

pub fn action_desc(a: &Action, s: &State, ctx: &Ctx) -> [u8; ACT_BYTES] {
    let p = s.to_act();
    let token = |unit: u8| match ctx.slot_of[p as usize].get(unit as usize) {
        Some(&k) if k >= 0 => p * NSLOT as u8 + k as u8,
        _ => NONE,
    };
    let h = a.hexes();
    [a.kind() as u8, token(action_coin(a, s)), token(a.recruited()), h[0], h[1], h[2]]
}

pub fn mirror_desc(d: &mut [u8]) {
    for c in &mut d[COIN_PICKS] {
        if (*c as usize) < NTYPE {
            *c = ((*c as usize + NSLOT) % NTYPE) as u8;
        }
    }
    for h in &mut d[HEX_PICKS] {
        if (*h as usize) < N_HEXES {
            *h = mirror_hex(*h as usize) as u8;
        }
    }
}

#[derive(Clone)]
pub struct Query {
    pub rows: Vec<u8>,
    pub desc: Vec<u8>,
    pub off: Vec<u32>,
}

impl Default for Query {
    fn default() -> Query {
        Query { rows: Vec::new(), desc: Vec::new(), off: vec![0] }
    }
}

impl Query {
    pub fn push(&mut self, s: &State, ctx: &Ctx, acts: &[Action]) {
        let at = self.rows.len();
        self.rows.resize(at + ROW_BYTES, 0);
        pack_row(s, ctx, &mut self.rows[at..]);
        for a in acts {
            self.desc.extend(action_desc(a, s, ctx));
        }
        self.off.push((self.desc.len() / ACT_BYTES) as u32);
    }

    pub fn row(&self, i: usize) -> (&[u8], &[u8]) {
        let a = self.actions(i);
        (&self.rows[i * ROW_BYTES..(i + 1) * ROW_BYTES], &self.desc[a.start * ACT_BYTES..a.end * ACT_BYTES])
    }

    pub fn push_row(&mut self, (row, desc): (&[u8], &[u8])) {
        self.rows.extend_from_slice(row);
        self.desc.extend_from_slice(desc);
        self.off.push((self.desc.len() / ACT_BYTES) as u32);
    }

    pub fn append(&mut self, o: &Query) {
        let base = (self.desc.len() / ACT_BYTES) as u32;
        self.rows.extend_from_slice(&o.rows);
        self.desc.extend_from_slice(&o.desc);
        self.off.extend(o.off[1..].iter().map(|x| x + base));
    }

    pub fn clear(&mut self) {
        self.rows.clear();
        self.desc.clear();
        self.off.truncate(1);
    }

    pub fn len(&self) -> usize {
        self.off.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn actions(&self, i: usize) -> Range<usize> {
        self.off[i] as usize..self.off[i + 1] as usize
    }

    pub fn tiles(&self, rows: usize, acts: usize) -> Vec<Range<usize>> {
        let (n, mut tiles) = (self.len(), Vec::new());
        let mut r0 = 0;
        while r0 < n {
            let mut r1 = r0 + 1;
            while r1 < n && r1 - r0 < rows && (self.off[r1 + 1] - self.off[r0]) as usize <= acts {
                r1 += 1;
            }
            assert!((self.off[r1] - self.off[r0]) as usize <= acts, "one position offers more moves than a tile holds");
            tiles.push(r0..r1);
            r0 = r1;
        }
        tiles
    }

    pub fn face_movers(&mut self) {
        for r in 0..self.len() {
            let (at, acts) = (r * ROW_BYTES..(r + 1) * ROW_BYTES, self.actions(r));
            if self.rows[at.start + ROW_TO_ACT] == 1 {
                let src = <[u8; ROW_BYTES]>::try_from(&self.rows[at.clone()]).expect("a row is ROW_BYTES long");
                mirror_row(&src, &mut self.rows[at]);
                self.desc[acts.start * ACT_BYTES..acts.end * ACT_BYTES].chunks_exact_mut(ACT_BYTES).for_each(mirror_desc);
            }
        }
    }
}

pub fn uniform(q: &Query) -> Vec<f32> {
    (0..q.len())
        .flat_map(|i| {
            let n = q.actions(i).len();
            std::iter::repeat_n(1.0 / n as f32, n)
        })
        .collect()
}

#[derive(Default)]
pub struct Answer {
    pub value: Vec<f32>,
    pub prior: Vec<f32>,
}

pub trait Oracle: Send + Sync {
    fn eval(&self, q: &Query) -> Answer;
}

#[derive(Default)]
pub struct Keyed(u64);

impl Hasher for Keyed {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("tree keys are hashed already")
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
}

fn mix(h: u64, x: u64) -> u64 {
    let v = (h.rotate_left(23) ^ x).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    v ^ (v >> 31)
}

fn private(s: &State, p: u8, ctx: &Ctx) -> u64 {
    let z = &s.zones[p as usize];
    ctx.slots[p as usize].iter().fold(0, |k, &u| {
        let u = u as usize;
        k << 12 | (z[Z_HAND][u] as u64) << 8 | (z[Z_FACEDOWN][u] as u64) << 4 | z[Z_INFLIGHT][u] as u64
    })
}

struct Node {
    first: u32,
    len: u32,
    player: u8,
    value: f32,
    stale: bool,
}

struct Edge {
    action: Action,
    prior: f32,
    n: u32,
    w: f32,
    pending: u32,
    pi: f32,
}

struct Step {
    node: u32,
    edge: u32,
}

#[derive(Default)]
struct Batch {
    q: Query,
    path: Vec<Step>,
    fresh: Vec<(u32, Range<usize>, Range<usize>)>,
    acts: Vec<Action>,
    legal: Vec<Action>,
}

enum Leaf {
    Ended,
    Fresh(u32),
    Busy,
}

fn halving(m: usize, sims: u32) -> Vec<(usize, u32)> {
    let rounds = (usize::BITS - (m - 1).leading_zeros()).max(1);
    let (mut k, mut left, mut phases) = (m, sims, Vec::new());
    while left > 0 {
        let n = (k as u32 * (sims / (rounds * k as u32)).max(1)).min(left);
        phases.push((k, n));
        left -= n;
        k = (k / 2).max(2);
    }
    phases
}

struct Tree {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    index: HashMap<u64, u32, BuildHasherDefault<Keyed>>,
    scale: f32,
}

impl Tree {
    fn new(search: Search) -> Tree {
        let nodes = 1 + search.sims as usize;
        Tree { nodes: Vec::with_capacity(nodes), edges: Vec::new(), index: HashMap::with_capacity_and_hasher(nodes, Default::default()), scale: search.scale }
    }

    fn node(&mut self, player: u8) -> u32 {
        self.nodes.push(Node { first: 0, len: 0, player, value: 0.0, stale: true });
        (self.nodes.len() - 1) as u32
    }

    fn expand(&mut self, node: u32, acts: &[Action], prior: &[f32], value: f32) {
        let n = &mut self.nodes[node as usize];
        n.first = self.edges.len() as u32;
        n.len = acts.len() as u32;
        n.value = value;
        self.edges.extend(acts.iter().zip(prior).map(|(&action, &prior)| Edge { action, prior, n: 0, w: 0.0, pending: 0, pi: 0.0 }));
    }

    fn span(&self, node: u32) -> Range<usize> {
        let n = &self.nodes[node as usize];
        n.first as usize..(n.first + n.len) as usize
    }

    fn edges(&self, node: u32) -> &[Edge] {
        &self.edges[self.span(node)]
    }

    fn completed(&self, node: u32) -> impl Fn(&Edge) -> f32 {
        let edges = self.edges(node);
        let visits = edges.iter().map(|e| e.n).sum::<u32>() as f32;
        let (pq, p) = edges.iter().filter(|e| e.n > 0).fold((0.0, 0.0), |(pq, p), e| {
            let prior = e.prior.max(f32::MIN_POSITIVE);
            (pq + prior * e.w / e.n as f32, p + prior)
        });
        let mean = if p > 0.0 { pq / p } else { 0.0 };
        let mix = (self.nodes[node as usize].value + visits * mean) / (1.0 + visits);
        move |e| if e.n > 0 { e.w / e.n as f32 } else { mix }
    }

    fn sigma(&self, node: u32) -> impl Fn(&Edge) -> f32 {
        let most = self.edges(node).iter().map(|e| e.n).max().unwrap_or(0) as f32;
        let (q, scale) = (self.completed(node), (C_VISIT + most) * self.scale);
        move |e| scale * (q(e) + 1.0) / 2.0
    }

    fn logits(&self, node: u32) -> impl Iterator<Item = f32> + '_ {
        let sigma = self.sigma(node);
        self.edges(node).iter().map(move |e| e.prior.ln() + sigma(e))
    }

    fn improved(&mut self, node: u32) -> &[Edge] {
        let span = self.span(node);
        if std::mem::take(&mut self.nodes[node as usize].stale) {
            let sigma = self.sigma(node);
            let edges = &mut self.edges[span.clone()];
            edges.iter_mut().for_each(|e| e.pi = e.prior.ln() + sigma(e));
            let top = edges.iter().map(|e| e.pi).fold(f32::NEG_INFINITY, f32::max);
            edges.iter_mut().for_each(|e| e.pi = (e.pi - top).exp());
            let total: f32 = edges.iter().map(|e| e.pi).sum();
            edges.iter_mut().for_each(|e| e.pi /= total);
        }
        &self.edges[span]
    }

    fn select(&mut self, node: u32) -> u32 {
        let first = self.nodes[node as usize].first;
        let edges = self.improved(node);
        let total = (1 + edges.iter().map(|e| e.n + e.pending).sum::<u32>()) as f32;
        let score = edges.iter().map(|e| e.pi - (e.n + e.pending) as f32 / total);
        let best = score.enumerate().max_by(|a, b| a.1.total_cmp(&b.1)).expect("an expanded node has an edge").0;
        first + best as u32
    }

    fn descend(&mut self, w: &mut State, ctx: &Ctx, rng: &mut Rng, mut edge: u32, path: &mut Vec<Step>) -> Leaf {
        let (mut node, mut h) = (0, 0u64);
        loop {
            let e = &mut self.edges[edge as usize];
            e.pending += 1;
            path.push(Step { node, edge });
            let action = e.action;
            debug_assert!(w.legal_actions().contains(&action), "an information set offers the same moves in every world");
            h = mix(h, obs_key(&action) as u64);
            w.apply_inplace(action);
            while w.is_chance() {
                resolve_chance(w, w.to_act(), rng);
            }
            if warchest::finished(w) {
                return Leaf::Ended;
            }
            let p = w.to_act();
            let key = mix(mix(h, p as u64), private(w, p, ctx));
            node = match self.index.get(&key) {
                Some(&node) if self.nodes[node as usize].len > 0 => node,
                Some(_) => return Leaf::Busy,
                None => {
                    let fresh = self.node(p);
                    self.index.insert(key, fresh);
                    return Leaf::Fresh(fresh);
                }
            };
            edge = self.select(node);
        }
    }

    fn backup(&mut self, path: &[Step], white: f32) {
        for s in path {
            let node = &mut self.nodes[s.node as usize];
            node.stale = true;
            let sign = if node.player == 0 { 1.0 } else { -1.0 };
            let e = &mut self.edges[s.edge as usize];
            e.pending -= 1;
            e.n += 1;
            e.w += sign * white;
        }
    }

    fn undo(&mut self, path: &[Step]) {
        for s in path {
            self.edges[s.edge as usize].pending -= 1;
        }
    }

    fn simulate(&mut self, batch: &mut Batch, oracle: &dyn Oracle, root: &State, ctx: &Ctx, opp: &Belief, rng: &mut Rng, visits: &[u32]) {
        let Batch { q, path, fresh, acts, legal } = batch;
        let mut next = 0;
        while next < visits.len() {
            q.clear();
            while next < visits.len() {
                let (mut w, start) = (*root, path.len());
                opp.deal(&mut w, rng);
                match self.descend(&mut w, ctx, rng, visits[next], path) {
                    Leaf::Ended => {
                        self.backup(&path[start..], warchest::score(&w, 0));
                        path.truncate(start);
                    }
                    Leaf::Busy => {
                        self.undo(&path[start..]);
                        path.truncate(start);
                        break;
                    }
                    Leaf::Fresh(node) => {
                        w.legal_actions_into(legal);
                        q.push(&w, ctx, legal);
                        fresh.push((node, start..path.len(), acts.len()..acts.len() + legal.len()));
                        acts.extend_from_slice(legal);
                    }
                }
                next += 1;
            }
            if q.is_empty() {
                continue;
            }
            let answer = oracle.eval(q);
            for (i, (node, steps, moves)) in fresh.drain(..).enumerate() {
                let v = answer.value[i];
                self.expand(node, &acts[moves.clone()], &answer.prior[moves], v);
                self.backup(&path[steps], if self.nodes[node as usize].player == 0 { v } else { -v });
            }
            path.clear();
            acts.clear();
        }
    }
}

pub struct Choice {
    pub probs: Vec<f32>,
    pub pick: usize,
    pub value: f32,
}

#[derive(Clone, Copy, Deserialize)]
pub struct Search {
    pub sims: u32,
    #[serde(default = "eight")]
    pub considered: usize,
    #[serde(default = "yes")]
    pub noise: bool,
    #[serde(default = "one")]
    pub scale: f32,
}

fn eight() -> usize {
    8
}

fn yes() -> bool {
    true
}

fn one() -> f32 {
    1.0
}

impl Search {
    pub fn full(sims: u32) -> Search {
        Search { sims, considered: 8, noise: true, scale: 1.0 }
    }

    pub fn fast(sims: u32) -> Search {
        Search { sims, considered: 4, noise: false, scale: 1.0 }
    }
}

pub fn think(oracle: &dyn Oracle, search: Search, root: &State, ctx: &Ctx, opp: &Belief, rng: &mut Rng) -> Choice {
    let (acts, mut batch) = (root.legal_actions(), Batch::default());
    batch.q.push(root, ctx, &acts);
    let answer = oracle.eval(&batch.q);
    let Search { sims, considered, noise: noisy, .. } = search;
    let mut tree = Tree::new(search);
    let node = tree.node(root.to_act());
    tree.expand(node, &acts, &answer.prior, answer.value[0]);
    let gumbel: Vec<f32> = acts.iter().map(|_| if noisy { rng.gumbel() } else { 0.0 }).collect();
    let score = |tree: &Tree| -> Vec<f32> { tree.logits(0).zip(&gumbel).map(|(l, g)| g + l).collect() };
    let mut alive: Vec<u32> = (0..acts.len() as u32).collect();
    for (k, n) in halving(acts.len().min(considered), sims) {
        let s = score(&tree);
        alive.sort_by(|&a, &b| s[b as usize].total_cmp(&s[a as usize]));
        alive.truncate(k);
        let visits: Vec<u32> = (0..n as usize).map(|j| alive[j % k]).collect();
        tree.simulate(&mut batch, oracle, root, ctx, opp, rng, &visits);
    }
    let (s, edges) = (score(&tree), tree.edges(0));
    let most = edges.iter().map(|e| e.n).max().expect("a root has a move");
    let pick = (0..edges.len())
        .filter(|&i| edges[i].n == most)
        .max_by(|&a, &b| s[a].total_cmp(&s[b]))
        .expect("a most visited move exists");
    let q = tree.completed(0);
    let edges = tree.improved(0);
    let probs = edges.iter().map(|e| e.pi).collect();
    let value = edges.iter().map(|e| e.pi * q(e)).sum();
    Choice { probs, pick, value }
}

pub struct Flat;

impl Oracle for Flat {
    fn eval(&self, q: &Query) -> Answer {
        Answer { value: vec![0.0; q.len()], prior: uniform(q) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::greedy;
    use warchest::rng::cumulative;

    fn root(prior: &[f32], value: f32, visits: &[(u32, f32)]) -> Tree {
        let mut tree = Tree::new(Search::full(0));
        let acts: Vec<Action> = (0..prior.len() as u8).map(|coin| Action::Pass { coin }).collect();
        let node = tree.node(0);
        tree.expand(node, &acts, prior, value);
        for (e, &(n, w)) in tree.edges.iter_mut().zip(visits) {
            (e.n, e.w) = (n, w);
        }
        tree
    }

    fn sigma(tree: &Tree) -> Vec<f32> {
        tree.edges(0).iter().map(tree.sigma(0)).collect()
    }

    fn improved(mut tree: Tree) -> Vec<f32> {
        tree.improved(0).iter().map(|e| e.pi).collect()
    }

    fn close(got: &[f32], want: &[f32]) {
        assert!(got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-4), "{got:?} is not {want:?}");
    }

    #[test]
    fn sequential_halving_spends_the_budget_like_mctx() {
        assert_eq!(halving(16, 32), [(16, 16), (8, 8), (4, 8)]);
        assert_eq!(halving(16, 200), [(16, 48), (8, 48), (4, 48), (2, 50), (2, 6)]);
        assert_eq!(halving(16, 8), [(16, 8)]);
        assert_eq!(halving(3, 32), [(3, 15), (2, 16), (2, 1)]);
        assert_eq!(halving(2, 32), [(2, 32)]);
        assert_eq!(halving(1, 5), [(1, 5)]);
    }

    #[test]
    fn an_unvisited_move_completes_with_the_mixed_value() {
        let tree = root(&[0.5, 0.3, 0.2], 0.2, &[(2, 1.0), (1, -1.0), (0, 0.0)]);
        let mix = 0.003125;
        close(&sigma(&tree), &[52.0 * 0.75, 0.0, 52.0 * (1.0 + mix) / 2.0]);
        close(&sigma(&root(&[0.5, 0.5], -0.4, &[])), &[50.0 * 0.3, 50.0 * 0.3]);
    }

    #[test]
    fn the_improved_policy_follows_the_better_q() {
        close(&improved(root(&[0.8, 0.2], 0.0, &[(1, 0.0), (1, 0.1)])), &[0.2380, 0.7620]);
        close(&improved(root(&[0.8, 0.2], 0.0, &[])), &[0.8, 0.2]);
        assert!(improved(root(&[0.8, 0.2], 0.0, &[(1, -1.0), (1, 1.0)]))[1] > 0.99);
    }

    #[test]
    fn a_pending_visit_turns_the_next_walk_away_without_changing_the_policy() {
        let mut tree = root(&[0.6, 0.4], 0.0, &[]);
        tree.edges[0].pending = 1;
        assert_eq!(tree.select(0), 1);
        close(&improved(tree), &[0.6, 0.4]);
    }

    #[test]
    fn thirty_two_simulations_find_the_only_winning_move() {
        let (mut rng, mut found) = (Rng::new(1), 0);
        while found < 10 {
            let mut s = warchest::arena::game(&mut rng);
            let ctx = Ctx::new(&s);
            while !warchest::finished(&s) {
                let acts = s.legal_actions();
                if s.is_chance() {
                    s.apply_inplace(acts[rng.below(acts.len())]);
                    continue;
                }
                let me = s.to_act();
                let wins: Vec<bool> = acts.iter().map(|&a| s.apply(a).winner() == Some(me)).collect();
                if acts.len() <= 8 && wins.iter().filter(|&&w| w).count() == 1 {
                    let c = think(&Flat, Search::full(32), &s, &ctx, &Belief::new(&s, 1 - me), &mut rng);
                    assert!(wins[c.pick], "the search missed the win");
                    assert!(c.probs[c.pick] > 0.9, "the target gives the win only {}", c.probs[c.pick]);
                    assert!(c.value > 0.5, "the root value of a won position is only {}", c.value);
                    found += 1;
                }
                s.apply_inplace(acts[rng.pick(&cumulative(greedy(&s, &acts).iter().map(|&p| p as f64)))]);
            }
        }
    }
}
