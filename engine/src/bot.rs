use std::sync::Arc;

use warchest::actions::Action;
use warchest::arena::{Draft, Obs};
use warchest::actions::obs_key;
use crate::pbs::{ways, Belief, Ctx};
use crate::policy;
use warchest::rng::{cumulative, Rng};
use crate::search::{think, Choice, Oracle, Query, Search};
use warchest::state::State;

const EPS: f64 = 0.3;

fn read(net: &dyn Oracle, ctx: &Ctx, worlds: &[(State, Vec<Action>)]) -> Vec<f64> {
    let mut q = Query::default();
    for (s, acts) in worlds {
        q.push(s, ctx, acts);
    }
    let n = worlds.iter().flat_map(|(_, acts)| std::iter::repeat_n(acts.len() as f64, acts.len()));
    net.eval(&q).prior.into_iter().zip(n).map(|(pi, n)| (1.0 - EPS) * pi as f64 + EPS / n).collect()
}

pub enum Brain {
    Random,
    Greedy,
    Search { search: Search, oracle: Arc<dyn Oracle> },
}

impl Brain {
    pub fn net(&self) -> Option<&dyn Oracle> {
        match self {
            Brain::Search { oracle, .. } => Some(oracle.as_ref()),
            _ => None,
        }
    }

    fn policy(&self, s: &State, ctx: &Ctx, opp: &Belief, acts: &[Action], rng: &mut Rng) -> Choice {
        let probs = match self {
            _ if acts.len() == 1 => return Choice { probs: vec![1.0], pick: 0, value: 0.0 },
            Brain::Random => vec![1.0 / acts.len() as f32; acts.len()],
            Brain::Greedy => policy::greedy(s, acts),
            Brain::Search { search, oracle } => return think(oracle.as_ref(), *search, s, ctx, opp, rng),
        };
        let pick = rng.pick(&cumulative(probs.iter().map(|&p| p as f64)));
        Choice { probs, pick, value: 0.0 }
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
            opp: Belief::new(&s, 1 - seat),
            s,
            seat,
            rng: Rng::new(seed),
        })
    }

    pub fn observe(&mut self, obs: &Obs, net: Option<&dyn Oracle>) -> Result<(), String> {
        match *obs {
            Obs::Draw { player, code } => self.drew(player, code),
            Obs::Act { player, key } if player == self.seat => self.acted(key),
            Obs::Act { player, key } => self.watched(player, key, net),
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
        self.s = self.opp.update(&self.s, |_| true, ways).ok_or("the opponent cannot draw")?;
        Ok(())
    }

    fn acted(&mut self, key: u32) -> Result<(), String> {
        let action = Action::decode(key)
            .filter(|a| self.to_move(self.seat) && self.s.legal_actions().contains(a))
            .ok_or_else(|| format!("own move {} is not legal here", key))?;
        self.s.apply_inplace(action);
        Ok(())
    }

    fn watched(&mut self, player: u8, key: u32, net: Option<&dyn Oracle>) -> Result<(), String> {
        if !self.to_move(player) {
            return Err(format!("player {} is not to act", player));
        }
        let ctx = &self.ctx;
        let likelihood = |worlds: &[(State, Vec<Action>)]| match net {
            Some(net) => read(net, ctx, worlds),
            None => ways(worlds),
        };
        self.s = self
            .opp
            .update(&self.s, |a| obs_key(a) == key, likelihood)
            .ok_or_else(|| format!("observation {} is unreachable from this state", key))?;
        Ok(())
    }

    fn to_move(&self, player: u8) -> bool {
        !self.s.is_terminal() && !self.s.is_chance() && self.s.to_act() == player
    }

    pub fn think(&mut self, brain: &Brain) -> Result<(Vec<Action>, Choice), String> {
        if !self.to_move(self.seat) {
            return Err("the bot was asked to move out of turn".into());
        }
        let acts = self.s.legal_actions();
        let choice = brain.policy(&self.s, &self.ctx, &self.opp, &acts, &mut self.rng);
        Ok((acts, choice))
    }

    pub fn play(&mut self, action: Action) {
        self.s.apply_inplace(action);
    }

    pub fn decide(&mut self, brain: &Brain) -> Result<u32, String> {
        let (acts, c) = self.think(brain)?;
        self.play(acts[c.pick]);
        Ok(acts[c.pick].encode())
    }

    pub fn query(&self, acts: &[Action], q: &mut Query) {
        q.push(&self.s, &self.ctx, acts);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use warchest::arena::{Done, Table};
    use crate::search::Flat;

    fn draft() -> Draft {
        Draft {
            white: vec![17, 12, 4, 9],
            black: vec![1, 3, 8, 16],
            first: 0,
        }
    }

    #[test]
    fn a_seat_is_told_nothing_private_and_follows_the_game() {
        for net in [None, Some(&Flat as &dyn Oracle)] {
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
                            seats[bot].observe(obs, net).unwrap();
                        }
                        done.push(Done { id: ask.id, action: seats[bot].decide(&brain).unwrap() });
                    }
                    table.accept(bot, done).unwrap();
                }
            }
            assert!(checked > 100, "the game was too short to have tested anything");
        }
    }
}
