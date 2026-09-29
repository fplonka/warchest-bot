use std::sync::Arc;

use crate::actions::Action;
use crate::arena::{Draft, Obs};
use crate::pbs::{deal, obs_key, unseen, Ctx};
use crate::policy;
use crate::rng::Rng;
use crate::search::{think, Oracle, Query, Search};
use crate::state::State;

pub enum Brain {
    Random,
    Greedy,
    Search { search: Search, oracle: Arc<dyn Oracle> },
}

impl Brain {
    #[cfg(feature = "gpu")]
    pub fn load(packed: &crate::packed::Packed, devices: &[usize]) -> Result<Brain, String> {
        let m: crate::packed::Mind = serde_json::from_value(packed.manifest.search.clone()).map_err(|e| e.to_string())?;
        match m.mind.as_str() {
            "greedy" => Ok(Brain::Greedy),
            "search" => {
                let weights = packed.dir.join(&packed.manifest.weights);
                let net = crate::net::Net::load_bin(weights.to_str().ok_or("invalid weights path")?)
                    .map_err(|e| format!("{}: {e}", weights.display()))?;
                let oracle = crate::cuda::Gpu::new(devices, &net)?;
                Ok(Brain::Search { search: Search::Full(m.sims), oracle: Arc::new(oracle) })
            }
            other => Err(format!("unknown mind {other}")),
        }
    }

    fn policy(&self, s: &State, ctx: &Ctx, acts: &[Action], rng: &mut Rng) -> (Vec<f32>, usize) {
        let probs = match self {
            _ if acts.len() == 1 => return (vec![1.0], 0),
            Brain::Random => vec![1.0 / acts.len() as f32; acts.len()],
            Brain::Greedy => policy::greedy(s, acts),
            Brain::Search { search, oracle } => return think(oracle.as_ref(), *search, s, ctx, rng),
        };
        let weights: Vec<f64> = probs.iter().map(|&p| p as f64).collect();
        let pick = rng.weighted_index(&weights);
        (probs, pick)
    }
}

pub struct Session {
    s: State,
    ctx: Ctx,
    seat: u8,
    rng: Rng,
}

impl Session {
    pub fn new(draft: &Draft, seat: u8, seed: u64) -> Result<Session, String> {
        let s = draft.state()?;
        Ok(Session {
            ctx: Ctx::new(&s),
            s,
            seat,
            rng: Rng::new(seed),
        })
    }

    pub fn observe(&mut self, obs: &Obs) -> Result<(), String> {
        match *obs {
            Obs::Draw { player, code } => self.drew(player, code),
            Obs::Act { player, key } if player == self.seat => self.acted(key),
            Obs::Act { player, key } => self.watched(player, key),
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
        self.s.apply_inplace(self.s.legal_actions()[0]);
        Ok(())
    }

    fn acted(&mut self, key: u32) -> Result<(), String> {
        let action = Action::decode(key)
            .filter(|a| self.to_move(self.seat) && self.s.legal_actions().contains(a))
            .ok_or_else(|| format!("own move {} is not legal here", key))?;
        self.s.apply_inplace(action);
        Ok(())
    }

    fn watched(&mut self, player: u8, key: u32) -> Result<(), String> {
        if !self.to_move(player) {
            return Err(format!("player {} is not to act", player));
        }
        let coins = unseen(&self.s, player);
        let fronted = (0..coins.len()).map(|i| {
            let (mut c, mut w) = (coins.clone(), self.s);
            c.swap(0, i);
            deal(&mut w, player, &c);
            w
        });
        let next = std::iter::once(self.s)
            .chain(fronted)
            .find_map(|w| w.legal_actions().into_iter().find(|a| obs_key(a) == key).map(|a| w.apply(a)));
        self.s = next.ok_or_else(|| format!("observation {} is unreachable from this state", key))?;
        Ok(())
    }

    fn to_move(&self, player: u8) -> bool {
        !self.s.is_terminal() && !self.s.is_chance() && self.s.to_act() == player
    }

    pub fn think(&mut self, brain: &Brain) -> Result<(Vec<Action>, Vec<f32>, usize), String> {
        if !self.to_move(self.seat) {
            return Err("the bot was asked to move out of turn".into());
        }
        let acts = self.s.legal_actions();
        let (probs, pick) = brain.policy(&self.s, &self.ctx, &acts, &mut self.rng);
        Ok((acts, probs, pick))
    }

    pub fn play(&mut self, action: Action) {
        self.s.apply_inplace(action);
    }

    pub fn decide(&mut self, brain: &Brain) -> Result<u32, String> {
        let (acts, _, pick) = self.think(brain)?;
        self.play(acts[pick]);
        Ok(acts[pick].encode())
    }

    pub fn query(&self, acts: &[Action], q: &mut Query) {
        q.push(&self.s, &self.ctx, acts);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{Done, Table};

    fn draft() -> Draft {
        Draft {
            white: vec![17, 12, 4, 9],
            black: vec![1, 3, 8, 16],
            first: 0,
        }
    }

    #[test]
    fn a_seat_is_told_nothing_private_and_follows_the_game() {
        let brain = Brain::Random;
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
                        seats[bot].observe(obs).unwrap();
                    }
                    done.push(Done { id: ask.id, action: seats[bot].decide(&brain).unwrap() });
                }
                table.accept(bot, done).unwrap();
            }
        }
        assert!(checked > 100, "the game was too short to have tested anything");
    }
}
