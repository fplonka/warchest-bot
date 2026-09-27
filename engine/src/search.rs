use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use serde::{Deserialize, Serialize};

use crate::actions::Action;
use crate::board::NONE;
use crate::pbs::{obs_key, pack_row, set_config, true_config, Belief, Ctx, NSLOT, ROW_BYTES};
use crate::rng::Rng;
use crate::selfplay::resolve_chance;
use crate::state::State;
use crate::units::{ENSIGN, MARSHAL, ROYAL_COIN};

pub const ACT_BYTES: usize = 6;
const NOISE_MASS: f64 = 10.0;

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

pub struct Answer {
    pub value: Vec<f32>,
    pub prior: Vec<f32>,
}

pub trait Oracle: Send + Sync {
    fn eval(&self, q: Query) -> Answer;
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Search {
    pub sims: u32,
    pub batch: u32,
    pub puct: f32,
    pub explore: f32,
}

#[derive(Default)]
struct Keyed(u64);

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

struct Node {
    first: u32,
    len: u32,
    n: u32,
    player: u8,
}

struct Edge {
    action: Action,
    prior: f32,
    n: u32,
    w: f32,
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

#[derive(Default)]
struct Tree {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    index: HashMap<u64, u32, BuildHasherDefault<Keyed>>,
}

impl Tree {
    fn node(&mut self, player: u8) -> u32 {
        self.nodes.push(Node { first: 0, len: 0, n: 0, player });
        (self.nodes.len() - 1) as u32
    }

    fn expand(&mut self, node: u32, acts: &[Action], prior: &[f32]) {
        let n = &mut self.nodes[node as usize];
        n.first = self.edges.len() as u32;
        n.len = acts.len() as u32;
        self.edges.extend(acts.iter().zip(prior).map(|(&action, &prior)| Edge { action, prior, n: 0, w: 0.0 }));
    }

    fn perturb_root(&mut self, explore: f32, rng: &mut Rng) {
        let n = &self.nodes[0];
        let edges = &mut self.edges[n.first as usize..(n.first + n.len) as usize];
        let noise: Vec<f64> = edges.iter().map(|_| rng.gamma(NOISE_MASS / edges.len() as f64)).collect();
        let total: f64 = noise.iter().sum();
        for (e, g) in edges.iter_mut().zip(noise) {
            e.prior = (1.0 - explore) * e.prior + explore * (g / total) as f32;
        }
    }

    fn edges(&self, node: u32) -> &[Edge] {
        let n = &self.nodes[node as usize];
        &self.edges[n.first as usize..(n.first + n.len) as usize]
    }

    fn select(&self, node: u32, puct: f32) -> u32 {
        let n = &self.nodes[node as usize];
        let scale = puct * (n.n.max(1) as f32).sqrt();
        let score = |e: &Edge| {
            let q = if e.n > 0 { e.w / e.n as f32 } else { 0.0 };
            q + scale * e.prior / (1 + e.n) as f32
        };
        let edges = self.edges(node);
        let best = (0..edges.len())
            .max_by(|&a, &b| score(&edges[a]).total_cmp(&score(&edges[b])))
            .expect("an expanded node has an edge");
        n.first + best as u32
    }

    fn descend(&mut self, w: &mut State, ctx: &Ctx, rng: &mut Rng, puct: f32, path: &mut Vec<Step>) -> Leaf {
        let mut h = 0u64;
        loop {
            let p = w.to_act();
            let key = mix(mix(h, p as u64), true_config(w, p, ctx).key());
            let node = match self.index.get(&key) {
                Some(&node) if self.nodes[node as usize].len > 0 => node,
                Some(_) => return Leaf::Busy,
                None => {
                    let fresh = self.node(p);
                    self.index.insert(key, fresh);
                    return Leaf::Fresh(fresh);
                }
            };
            let edge = self.select(node, puct);
            self.nodes[node as usize].n += 1;
            let e = &mut self.edges[edge as usize];
            e.n += 1;
            e.w -= 1.0;
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
        }
    }

    fn backup(&mut self, path: &[Step], white: f32) {
        for s in path {
            let sign = if self.nodes[s.node as usize].player == 0 { 1.0 } else { -1.0 };
            self.edges[s.edge as usize].w += 1.0 + sign * white;
        }
    }

    fn undo(&mut self, path: &[Step]) {
        for s in path {
            self.nodes[s.node as usize].n -= 1;
            let e = &mut self.edges[s.edge as usize];
            e.n -= 1;
            e.w += 1.0;
        }
    }
}

impl Search {
    pub fn think(&self, oracle: &dyn Oracle, root: &State, ctx: &Ctx, opp: &Belief, rng: &mut Rng) -> Vec<f32> {
        let me = root.to_act();
        let mut tree = Tree::default();
        let (mut path, mut acts, mut legal, mut fresh) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut left = self.sims;
        while left > 0 {
            let mut q = Query::default();
            while fresh.len() < self.batch as usize && left > 0 {
                let (mut w, start) = (*root, path.len());
                set_config(&mut w, 1 - me, ctx, &opp.sample(rng));
                match tree.descend(&mut w, ctx, rng, self.puct, &mut path) {
                    Leaf::Ended => {
                        tree.backup(&path[start..], crate::score(&w, 0));
                        path.truncate(start);
                    }
                    Leaf::Busy => {
                        tree.undo(&path[start..]);
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
                left -= 1;
            }
            if q.is_empty() {
                continue;
            }
            let answer = oracle.eval(q);
            for (i, (node, steps, moves)) in fresh.drain(..).enumerate() {
                tree.expand(node, &acts[moves.clone()], &answer.prior[moves]);
                if node == 0 {
                    tree.perturb_root(self.explore, rng);
                }
                let v = answer.value[i];
                tree.backup(&path[steps], if tree.nodes[node as usize].player == 0 { v } else { -v });
            }
            path.clear();
            acts.clear();
        }
        let edges = tree.edges(0);
        let total = edges.iter().map(|e| e.n).sum::<u32>() as f32;
        edges.iter().map(|e| e.n as f32 / total).collect()
    }
}
