use std::sync::Arc;

use crate::actions::Action;
use crate::arena::{Draft, Obs};
use crate::pbs::{
    belief_after_draw, faceup_counts, posterior, reserve, set_config, uniform_belief, Belief, Config, Ctx,
};
use crate::policy;
use crate::rng::Rng;
use crate::search::{uniform, Oracle, Query, Search};
use crate::state::{Cont, State};

const SUPPORT: usize = 128;
const FLOOR: f32 = 1e-3;
const EPS: f32 = 0.05;

pub enum Mind {
    Random,
    Greedy,
    Search { search: Search, oracle: Arc<dyn Oracle> },
}

pub struct Brain {
    pub mind: Mind,
    pub temp: f32,
    pub cut: f32,
}

impl Brain {
    #[cfg(feature = "gpu")]
    pub fn load(packed: &crate::packed::Packed, devices: &[usize]) -> Result<Brain, String> {
        let m = &packed.manifest.search;
        let mind = match m.mind.as_str() {
            "greedy" => Mind::Greedy,
            "search" => {
                let weights = packed.dir.join(&packed.manifest.weights);
                let net = crate::net::Net::load_bin(weights.to_str().ok_or("invalid weights path")?)
                    .map_err(|e| format!("{}: {e}", weights.display()))?;
                let oracle = crate::cuda::Gpu::new(devices, &net)?;
                Mind::Search { search: m.search, oracle: Arc::new(oracle) }
            }
            other => return Err(format!("unknown mind {other}")),
        };
        Ok(Brain { mind, temp: m.temp, cut: m.cut })
    }

    fn policy(&self, s: &State, ctx: &Ctx, opp: &Belief, acts: &[Action], rng: &mut Rng) -> Vec<f32> {
        let raw = match &self.mind {
            _ if acts.len() == 1 => vec![1.0],
            Mind::Random => vec![1.0 / acts.len() as f32; acts.len()],
            Mind::Greedy => policy::greedy(s, acts),
            Mind::Search { search, oracle } => search.think(oracle.as_ref(), s, ctx, opp, rng),
        };
        let tempered: Vec<f32> = raw.iter().map(|p| p.powf(1.0 / self.temp)).collect();
        let floor = self.cut * tempered.iter().copied().fold(0.0, f32::max);
        let kept: Vec<f32> = tempered.into_iter().map(|p| if p < floor { 0.0 } else { p }).collect();
        let sum: f32 = kept.iter().sum();
        kept.into_iter().map(|p| p / sum).collect()
    }

    fn likelihood(&self, q: Query) -> Vec<f32> {
        let flat = uniform(&q);
        let prior = match &self.mind {
            Mind::Search { oracle, .. } if !q.is_empty() => oracle.eval(q).prior,
            _ => return flat,
        };
        flat.iter().zip(prior).map(|(u, p)| EPS * u + (1.0 - EPS) * p).collect()
    }
}

pub struct Session {
    s: State,
    ctx: Ctx,
    seat: u8,
    opp: Belief,
    rng: Rng,
}

impl Session {
    pub fn new(draft: &Draft, seat: u8, seed: u64) -> Result<Session, String> {
        let s = draft.state()?;
        Ok(Session {
            ctx: Ctx::new(&s),
            s,
            seat,
            opp: Belief::point(Config::default()),
            rng: Rng::new(seed),
        })
    }

    pub fn observe(&mut self, obs: &Obs, brain: &Brain) -> Result<(), String> {
        match *obs {
            Obs::Draw { player, code } => self.drew(player, code),
            Obs::Act { player, key } if player == self.seat => self.acted(key),
            Obs::Act { player, key } => self.watched(player, key, brain),
        }
    }

    fn drew(&mut self, player: u8, code: Option<u32>) -> Result<(), String> {
        if !self.s.is_chance() || self.s.to_act() != player {
            return Err(format!("player {} is not drawing", player));
        }
        if player == self.seat {
            let code = code.ok_or("this seat was not shown its own draw")?;
            let drawn = Action::decode(code).ok_or_else(|| format!("draw {} does not decode", code))?;
            self.s.apply_inplace(drawn);
            return Ok(());
        }
        let wp = matches!(self.s.pending(), Cont::WarriorPriestDraw { .. });
        let res = reserve(&self.s, player, &self.ctx);
        let faceup = faceup_counts(&self.s, player, &self.ctx);
        self.opp = belief_after_draw(&self.opp, &res, &faceup, wp);
        self.opp.prune(SUPPORT, FLOOR);
        self.s.apply_inplace(self.s.legal_actions()[0]);
        set_config(&mut self.s, player, &self.ctx, &self.opp.cfg[0]);
        Ok(())
    }

    fn acted(&mut self, key: u32) -> Result<(), String> {
        let action = Action::decode(key)
            .filter(|a| self.to_move(self.seat) && self.s.legal_actions().contains(a))
            .ok_or_else(|| format!("own move {} is not legal here", key))?;
        self.s.apply_inplace(action);
        Ok(())
    }

    fn watched(&mut self, player: u8, key: u32, brain: &Brain) -> Result<(), String> {
        if !self.to_move(player) {
            return Err(format!("player {} is not to act", player));
        }
        let likelihood = |q: Query| brain.likelihood(q);
        let (mut opp, mut next) = posterior(&self.s, &self.ctx, player, &self.opp, key, likelihood);
        if opp.is_empty() {
            let restart = uniform_belief(&self.s, &self.ctx, player);
            (opp, next) = posterior(&self.s, &self.ctx, player, &restart, key, likelihood);
        }
        if opp.is_empty() {
            return Err(format!("observation {} is unreachable from this state", key));
        }
        opp.prune(SUPPORT, FLOOR);
        self.opp = opp;
        self.s = next;
        set_config(&mut self.s, player, &self.ctx, &self.opp.cfg[0]);
        Ok(())
    }

    fn to_move(&self, player: u8) -> bool {
        !self.s.is_terminal() && !self.s.is_chance() && self.s.to_act() == player
    }

    pub fn think(&mut self, brain: &Brain) -> Result<(Vec<Action>, Vec<f32>), String> {
        if !self.to_move(self.seat) {
            return Err("the bot was asked to move out of turn".into());
        }
        let acts = self.s.legal_actions();
        let probs = brain.policy(&self.s, &self.ctx, &self.opp, &acts, &mut self.rng);
        Ok((acts, probs))
    }

    pub fn play(&mut self, acts: &[Action], probs: &[f32]) -> Action {
        let weights: Vec<f64> = probs.iter().map(|&p| p as f64).collect();
        let action = acts[self.rng.weighted_index(&weights)];
        self.s.apply_inplace(action);
        action
    }

    pub fn decide(&mut self, brain: &Brain) -> Result<u32, String> {
        let (acts, probs) = self.think(brain)?;
        Ok(self.play(&acts, &probs).encode())
    }

    pub fn query(&self, acts: &[Action], q: &mut Query) {
        q.push(&self.s, &self.ctx, acts);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{Done, Table};
    use crate::pbs::{obs_key, true_config};

    fn draft() -> Draft {
        Draft {
            white: vec![17, 12, 4, 9],
            black: vec![1, 3, 8, 16],
            first: 0,
        }
    }

    #[test]
    fn a_seat_is_told_nothing_private_and_follows_the_game() {
        let brain = Brain { mind: Mind::Random, temp: 1.0, cut: 0.0 };
        let mut table = Table::new();
        table.start(1, &draft(), [0, 1], 8191).unwrap();
        let mut seats = [
            Session::new(&draft(), 0, 1).unwrap(),
            Session::new(&draft(), 1, 2).unwrap(),
        ];
        let mut checked = 0;
        while table.live() > 0 {
            table.settle();
            for bot in 0..2 {
                let request = table.request(bot);
                let mut done = Vec::new();
                for ask in request.go {
                    for obs in &ask.obs {
                        match *obs {
                            Obs::Draw { player, code } => {
                                assert_eq!(code.is_some(), player == bot as u8, "seat {} saw a draw it should not", bot);
                            }
                            Obs::Act { player, key } => {
                                assert_ne!(player, bot as u8);
                                if let Some(action) = Action::decode(key) {
                                    assert_eq!(obs_key(&action), key, "an observation carried more than was seen");
                                }
                            }
                        }
                        checked += 1;
                        seats[bot].observe(obs, &brain).unwrap();
                    }
                    let other = 1 - seats[bot].seat;
                    assert!(seats[bot].opp.cfg.contains(&true_config(&seats[bot].s, other, &seats[bot].ctx)));
                    done.push(Done { id: ask.id, action: seats[bot].decide(&brain).unwrap() });
                }
                table.accept(bot, done).unwrap();
            }
        }
        assert!(checked > 100, "the game was too short to have tested anything");
    }
}
