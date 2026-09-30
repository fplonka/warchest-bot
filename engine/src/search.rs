use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate::actions::Action;
use crate::board::NONE;
use crate::pbs::{obs_key, pack_row, Belief, Ctx, NSLOT, ROW_BYTES};
use crate::rng::Rng;
use crate::selfplay::resolve_chance;
use crate::state::{State, Z_FACEDOWN, Z_HAND, Z_INFLIGHT};
use crate::units::{ENSIGN, MARSHAL, ROYAL_COIN};

pub const ACT_BYTES: usize = 6;
const C_VISIT: f32 = 50.0;
const C_SCALE: f32 = 1.0;

fn action_coin(a: &Action, s: &State) -> u8 {
    use Action::*;
    match *a {
        Deploy { unit, .. } | Bolster { unit, .. } => unit,
        ClaimInitiative { coin } | Recruit { coin, .. } | Pass { coin } => coin,
        TacRoyalGuard { .. } => ROYAL_COIN,
        TacEnsign { .. } => ENSIGN,
        TacMarshal { .. } => MARSHAL,
        Move { from, .. }
        | Control { from }
        | Attack { from, .. }
        | TacArcher { from, .. }
        | TacCavalryMove { from, .. }
        | TacCrossbow { from, .. }
        | TacLancer { from, .. }
        | TacLightCav { from, .. } => s.hex_type[from as usize],
        _ => NONE,
    }
}

pub fn action_desc(a: &Action, s: &State, ctx: &Ctx) -> [u8; ACT_BYTES] {
    let p = s.to_act();
    let token = |unit: u8| match ctx.slot_of[p as usize].get(unit as usize) {
        Some(&k) if k >= 0 => p * NSLOT as u8 + k as u8,
        _ => NONE,
    };
    let h = a.hexes();
    [a.kind() as u8, token(action_coin(a, s)), token(a.recruited()), h[0], h[1], h[2]]
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

    pub fn actions(&self, i: usize) -> std::ops::Range<usize> {
        self.off[i] as usize..self.off[i + 1] as usize
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
    fn eval(&self, q: Query) -> Answer;
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
}

struct Edge {
    action: Action,
    prior: f32,
    logit: f32,
    n: u32,
    w: f32,
    pending: u32,
}

struct Step {
    node: u32,
    edge: u32,
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

#[derive(Default)]
struct Tree {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    index: HashMap<u64, u32, BuildHasherDefault<Keyed>>,
}

impl Tree {
    fn node(&mut self, player: u8) -> u32 {
        self.nodes.push(Node { first: 0, len: 0, player, value: 0.0 });
        (self.nodes.len() - 1) as u32
    }

    fn expand(&mut self, node: u32, acts: &[Action], prior: &[f32], value: f32) {
        let n = &mut self.nodes[node as usize];
        n.first = self.edges.len() as u32;
        n.len = acts.len() as u32;
        n.value = value;
        self.edges.extend(acts.iter().zip(prior).map(|(&action, &prior)| Edge { action, prior, logit: prior.ln(), n: 0, w: 0.0, pending: 0 }));
    }

    fn edges(&self, node: u32) -> &[Edge] {
        let n = &self.nodes[node as usize];
        &self.edges[n.first as usize..(n.first + n.len) as usize]
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
        let (q, scale) = (self.completed(node), (C_VISIT + most) * C_SCALE);
        move |e| scale * (q(e) + 1.0) / 2.0
    }

    fn logits(&self, node: u32) -> impl Iterator<Item = f32> + '_ {
        let sigma = self.sigma(node);
        self.edges(node).iter().map(move |e| e.logit + sigma(e))
    }

    fn improved(&self, node: u32) -> Vec<f32> {
        let logits: Vec<f32> = self.logits(node).collect();
        let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = logits.iter().map(|l| (l - top).exp()).sum();
        logits.iter().map(|l| (l - top).exp() / z).collect()
    }

    fn select(&self, node: u32) -> u32 {
        let edges = self.edges(node);
        let total = (1 + edges.iter().map(|e| e.n + e.pending).sum::<u32>()) as f32;
        let top = self.logits(node).fold(f32::NEG_INFINITY, f32::max);
        let z: f32 = self.logits(node).map(|l| (l - top).exp()).sum();
        let score = self.logits(node).zip(edges).map(|(l, e)| (l - top).exp() / z - (e.n + e.pending) as f32 / total);
        let best = score.enumerate().max_by(|a, b| a.1.total_cmp(&b.1)).expect("an expanded node has an edge").0;
        self.nodes[node as usize].first + best as u32
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
            if crate::finished(w) {
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
            let sign = if self.nodes[s.node as usize].player == 0 { 1.0 } else { -1.0 };
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

    fn simulate(&mut self, oracle: &dyn Oracle, root: &State, ctx: &Ctx, opp: &Belief, rng: &mut Rng, visits: &[u32]) {
        let (mut path, mut acts, mut legal, mut fresh) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut next = 0;
        while next < visits.len() {
            let mut q = Query::default();
            while next < visits.len() {
                let (mut w, start) = (*root, path.len());
                opp.deal(&mut w, 1 - root.to_act(), rng);
                match self.descend(&mut w, ctx, rng, visits[next], &mut path) {
                    Leaf::Ended => {
                        self.backup(&path[start..], crate::score(&w, 0));
                        path.truncate(start);
                    }
                    Leaf::Busy => {
                        self.undo(&path[start..]);
                        path.truncate(start);
                        break;
                    }
                    Leaf::Fresh(node) => {
                        w.legal_actions_into(&mut legal);
                        q.push(&w, ctx, &legal);
                        fresh.push((node, start..path.len(), acts.len()..acts.len() + legal.len()));
                        acts.extend_from_slice(&legal);
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

#[derive(Clone, Copy)]
pub enum Search {
    Full(u32),
    Fast(u32),
}

pub fn think(oracle: &dyn Oracle, search: Search, root: &State, ctx: &Ctx, opp: &Belief, rng: &mut Rng) -> Choice {
    let acts = root.legal_actions();
    let mut q = Query::default();
    q.push(root, ctx, &acts);
    let answer = oracle.eval(q);
    let mut tree = Tree::default();
    let node = tree.node(root.to_act());
    tree.expand(node, &acts, &answer.prior, answer.value[0]);
    let (sims, considered, noisy) = match search {
        Search::Full(sims) => (sims, 8, true),
        Search::Fast(sims) => (sims, 4, false),
    };
    let gumbel: Vec<f32> = acts.iter().map(|_| if noisy { rng.gumbel() } else { 0.0 }).collect();
    let score = |tree: &Tree| -> Vec<f32> { tree.logits(0).zip(&gumbel).map(|(l, g)| g + l).collect() };
    let mut alive: Vec<u32> = (0..acts.len() as u32).collect();
    for (k, n) in halving(acts.len().min(considered), sims) {
        let s = score(&tree);
        alive.sort_by(|&a, &b| s[b as usize].total_cmp(&s[a as usize]));
        alive.truncate(k);
        let visits: Vec<u32> = (0..n as usize).map(|j| alive[j % k]).collect();
        tree.simulate(oracle, root, ctx, opp, rng, &visits);
    }
    let (s, edges) = (score(&tree), tree.edges(0));
    let most = edges.iter().map(|e| e.n).max().expect("a root has a move");
    let pick = (0..edges.len())
        .filter(|&i| edges[i].n == most)
        .max_by(|&a, &b| s[a].total_cmp(&s[b]))
        .expect("a most visited move exists");
    let (probs, q) = (tree.improved(0), tree.completed(0));
    let value = probs.iter().zip(edges).map(|(p, e)| p * q(e)).sum();
    Choice { probs, pick, value }
}

#[cfg(test)]
pub(crate) struct Flat;

#[cfg(test)]
impl Oracle for Flat {
    fn eval(&self, q: Query) -> Answer {
        Answer { value: vec![0.0; q.len()], prior: uniform(&q) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::greedy;

    fn root(prior: &[f32], value: f32, visits: &[(u32, f32)]) -> Tree {
        let mut tree = Tree::default();
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
        close(&root(&[0.8, 0.2], 0.0, &[(1, 0.0), (1, 0.1)]).improved(0), &[0.2380, 0.7620]);
        close(&root(&[0.8, 0.2], 0.0, &[]).improved(0), &[0.8, 0.2]);
        assert!(root(&[0.8, 0.2], 0.0, &[(1, -1.0), (1, 1.0)]).improved(0)[1] > 0.99);
    }

    #[test]
    fn a_pending_visit_turns_the_next_walk_away_without_changing_the_policy() {
        let mut tree = root(&[0.6, 0.4], 0.0, &[]);
        tree.edges[0].pending = 1;
        close(&tree.improved(0), &[0.6, 0.4]);
        assert_eq!(tree.select(0), 1);
    }

    #[test]
    fn thirty_two_simulations_find_the_only_winning_move() {
        let (mut rng, mut found) = (Rng::new(1), 0);
        while found < 10 {
            let mut s = crate::selfplay::game(&mut rng);
            let ctx = Ctx::new(&s);
            while !crate::finished(&s) {
                let acts = s.legal_actions();
                if s.is_chance() {
                    s.apply_inplace(acts[rng.below(acts.len())]);
                    continue;
                }
                let me = s.to_act();
                let wins: Vec<bool> = acts.iter().map(|&a| s.apply(a).winner() == Some(me)).collect();
                if acts.len() <= 8 && wins.iter().filter(|&&w| w).count() == 1 {
                    let c = think(&Flat, Search::Full(32), &s, &ctx, &Belief::new(&s, 1 - me), &mut rng);
                    assert!(wins[c.pick], "the search missed the win");
                    assert!(c.probs[c.pick] > 0.9, "the target gives the win only {}", c.probs[c.pick]);
                    assert!(c.value > 0.5, "the root value of a won position is only {}", c.value);
                    found += 1;
                }
                let w: Vec<f64> = greedy(&s, &acts).iter().map(|&p| p as f64).collect();
                s.apply_inplace(acts[rng.weighted_index(&w)]);
            }
        }
    }
}
