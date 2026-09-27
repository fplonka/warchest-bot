use std::ops::Range;

use crate::actions::Action;
use crate::board::{board, NONE, N_HEXES, N_LOCATIONS};
use crate::pbs::{action_legal, advance_config, obs_key, set_config, Belief, Config, Ctx};
use crate::rng::Rng;
use crate::search::{node_actions, Policy, Solver};
use crate::state::{State, Z_ELIM};

pub struct NodePolicy {
    pub acts: Vec<Action>,
    pub aslot: Vec<i8>,
    pub fdown: Vec<bool>,
    legal_off: Vec<u32>,
    legal_action: Vec<u32>,
    pub probs: Vec<f32>,
    pub q: Vec<f32>,
}

impl NodePolicy {
    pub fn frame(s: &State, ctx: &Ctx, player: u8, cfgs: &[Config]) -> NodePolicy {
        let (acts, aslot, fdown) = node_actions(s, player, ctx, cfgs);
        let na = acts.len();
        let mut legal_off = Vec::with_capacity(cfgs.len() + 1);
        let mut legal_action = Vec::new();
        legal_off.push(0);
        for c in cfgs {
            for a in 0..na {
                if action_legal(c, aslot[a]) {
                    legal_action.push(a as u32);
                }
            }
            legal_off.push(legal_action.len() as u32);
        }
        let cells = legal_action.len();
        NodePolicy {
            acts,
            aslot,
            fdown,
            legal_off,
            legal_action,
            probs: vec![0.0; cells],
            q: vec![f32::NAN; cells],
        }
    }

    #[inline]
    pub fn row(&self, c: usize) -> Range<usize> {
        self.legal_off[c] as usize..self.legal_off[c + 1] as usize
    }

    #[inline]
    pub fn action_at(&self, cell: usize) -> usize {
        self.legal_action[cell] as usize
    }

    pub fn to_replay(&self, player: u8, ctx: &Ctx) -> Policy {
        let ncfg = self.legal_off.len().saturating_sub(1);
        let mut out = Policy {
            acts: self
                .acts
                .iter()
                .enumerate()
                .map(|(a, act)| crate::search::action_desc(act, player, ctx, self.aslot[a]))
                .collect(),
            off: Vec::with_capacity(ncfg + 1),
            act: Vec::new(),
            p: Vec::new(),
            q: Vec::new(),
        };
        out.off.push(0);
        for ci in 0..ncfg {
            for cell in self.row(ci) {
                out.act.push(self.action_at(cell) as u16);
                out.p.push(self.probs[cell]);
                out.q.push(self.q[cell]);
            }
            out.off.push(out.act.len() as u32);
        }
        out
    }

    pub fn sample(&self, rng: &mut Rng, c: usize) -> usize {
        let row = self.row(c);
        let weights: Vec<f64> = self.probs[row.clone()].iter().map(|&x| x.max(0.0) as f64).collect();
        if weights.iter().all(|&w| w == 0.0) {
            return row.start + rng.below(row.len().max(1));
        }
        row.start + rng.weighted_index(&weights)
    }

    pub fn mix_uniform(&mut self, eps: f32) {
        if eps == 0.0 {
            return;
        }
        let n = self.legal_off.len().saturating_sub(1);
        for ci in 0..n {
            let row = self.row(ci);
            let k = row.len();
            if k == 0 {
                continue;
            }
            let u = 1.0 / k as f32;
            let keep = 1.0 - eps;
            for cell in row {
                self.probs[cell] = keep * self.probs[cell] + eps * u;
            }
        }
    }

    pub fn temper(&mut self, t: f32) {
        if t == 1.0 {
            return;
        }
        for ci in 0..self.legal_off.len() - 1 {
            let row = self.row(ci);
            let probs = &mut self.probs[row];
            let max = probs.iter().copied().fold(f32::MIN_POSITIVE, f32::max);
            probs.iter_mut().for_each(|p| *p = (*p / max).powf(1.0 / t));
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                probs.iter_mut().for_each(|p| *p /= sum);
            }
        }
    }

    pub fn cut(&mut self, frac: f32) {
        for ci in 0..self.legal_off.len() - 1 {
            let row = self.row(ci);
            let probs = &mut self.probs[row];
            let floor = frac * probs.iter().copied().fold(0.0, f32::max);
            probs.iter_mut().filter(|p| **p < floor).for_each(|p| *p = 0.0);
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                probs.iter_mut().for_each(|p| *p /= sum);
            }
        }
    }

    pub fn posterior(&self, prior: &Belief, obs: u32) -> Belief {
        let mut pairs: Vec<(Config, f32)> = Vec::new();
        for (ci, c) in prior.cfg.iter().enumerate() {
            for cell in self.row(ci) {
                let a = self.action_at(cell);
                if obs_key(&self.acts[a]) != obs {
                    continue;
                }
                if let Some(next) = advance_config(c, self.aslot[a], self.fdown[a]) {
                    pairs.push((next, prior.p[ci] * self.probs[cell]));
                }
            }
        }
        Belief::from_pairs(pairs)
    }

    pub fn cell_for(&self, obs: u32) -> Option<(usize, usize)> {
        (0..self.legal_off.len() - 1).find_map(|ci| {
            self.row(ci)
                .find(|&cell| obs_key(&self.acts[self.action_at(cell)]) == obs)
                .map(|cell| (ci, cell))
        })
    }
}

pub fn uniform(s: &State, ctx: &Ctx, player: u8, cfgs: &[Config]) -> NodePolicy {
    let mut np = NodePolicy::frame(s, ctx, player, cfgs);
    for ci in 0..cfgs.len() {
        let row = np.row(ci);
        let k = row.len() as f32;
        for cell in row {
            np.probs[cell] = 1.0 / k;
        }
    }
    np
}

pub fn root(sv: &Solver) -> NodePolicy {
    let n = &sv.nodes[0];
    let cfgs = n.cfgs[n.player as usize].as_ref();
    if n.legal_off.is_empty() {
        return uniform(&sv.nodes[0].state, &sv.ctx, n.player, cfgs);
    }
    let mut policy = NodePolicy {
        acts: n.acts.clone(),
        aslot: n.aslot.clone(),
        fdown: n.fdown.clone(),
        legal_off: n.legal_off.clone(),
        legal_action: n.legal_action.clone(),
        probs: vec![0.0; n.legal_action.len()],
        q: sv.root_q.clone(),
    };
    for config in 0..cfgs.len() {
        let row = policy.row(config);
        policy.probs[row].copy_from_slice(sv.root_strategy(config));
    }
    policy
}

pub fn eval_static(s: &State, p: u8) -> f32 {
    if s.is_terminal() {
        return s.utility(p as usize) * 1e6;
    }
    let b = board();
    let o = 1 - p;
    let mut sc = 12.0 * (s.markers_on_board(p) as f32 - s.markers_on_board(o) as f32);
    let (mut coins_p, mut coins_o) = (0.0f32, 0.0f32);
    for h in 0..N_HEXES {
        match s.hex_owner[h] {
            x if x == p => coins_p += s.hex_height[h] as f32,
            x if x == o => coins_o += s.hex_height[h] as f32,
            _ => {}
        }
    }
    sc += 1.5 * (coins_p - coins_o);
    let elim = |q: u8| -> f32 { s.zones[q as usize][Z_ELIM].iter().map(|&x| x as f32).sum() };
    sc += elim(o) - elim(p);
    let (mut cover_p, mut cover_o) = (0.0f32, 0.0f32);
    for li in 0..N_LOCATIONS {
        let l = b.location_hexes[li] as usize;
        let (mut bp, mut bo) = (7.0f32, 7.0f32);
        for h in 0..N_HEXES {
            let owner = s.hex_owner[h];
            if owner == NONE {
                continue;
            }
            let d = b.dist[h][l] as f32;
            if owner == p {
                bp = bp.min(d);
            } else {
                bo = bo.min(d);
            }
        }
        if s.loc_marker[l] != p {
            cover_p += bp;
        }
        if s.loc_marker[l] != o {
            cover_o += bo;
        }
    }
    sc += 0.5 * (cover_o - cover_p);
    sc
}

pub fn eval_squashed(s: &State, p: u8) -> f32 {
    (eval_static(s, p) / 25.0).tanh()
}

pub fn greedy(s: &State, ctx: &Ctx, player: u8, cfgs: &[Config], temp: f32) -> NodePolicy {
    let mut np = NodePolicy::frame(s, ctx, player, cfgs);
    let na = np.acts.len();
    let mut score = vec![f32::NEG_INFINITY; na];
    for a in 0..na {
        let Some(rep) = cfgs.iter().find(|c| action_legal(c, np.aslot[a])) else {
            continue;
        };
        let mut probe = *s;
        set_config(&mut probe, player, ctx, rep);
        probe.apply_inplace(np.acts[a]);
        score[a] = eval_static(&probe, player) / temp;
    }
    for ci in 0..cfgs.len() {
        let cells = np.row(ci);
        let best = cells
            .clone()
            .fold(f32::NEG_INFINITY, |best, cell| best.max(score[np.action_at(cell)]));
        let mut sum = 0.0f32;
        for cell in cells.clone() {
            let e = (score[np.action_at(cell)] - best).exp();
            np.probs[cell] = e;
            sum += e;
        }
        if sum > 0.0 {
            for cell in cells {
                np.probs[cell] /= sum;
            }
        }
    }
    np
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::Draft;
    use crate::pbs::true_config;

    #[test]
    fn a_tempered_mixed_opponent_can_play_any_legal_move() {
        let draft = Draft { white: vec![17, 12, 4, 9], black: vec![1, 3, 8, 16], first: 0 };
        let mut s = draft.state().unwrap();
        let ctx = Ctx::new(&s);
        while s.is_chance() {
            s.apply_inplace(s.legal_actions()[0]);
        }
        let cfgs = [true_config(&s, s.to_act(), &ctx)];
        let mut np = uniform(&s, &ctx, s.to_act(), &cfgs);
        let row = np.row(0);
        assert!(row.len() > 1);
        np.probs[row.clone()].fill(0.0);
        np.probs[row.start] = 1.0;
        let sharp = np.probs.clone();
        np.temper(1.0);
        np.mix_uniform(0.0);
        assert_eq!(np.probs, sharp);
        np.temper(3.0);
        np.mix_uniform(0.1);
        assert!(np.probs[row.clone()].iter().all(|&p| p > 0.0));
        assert!((np.probs[row.clone()].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        np.cut(1.0);
        assert_eq!(np.probs[row.start], 1.0);
        np.mix_uniform(0.5);
        np.temper(0.0);
        assert_eq!(np.probs[row.start], 1.0);
    }
}
