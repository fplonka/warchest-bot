use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use warchest::actions::Action;
use warchest::arena::{Ask, Draft, Obs};
use warchest::actions::obs_key;
use crate::pbs::{ways, Belief, Ctx};
use crate::policy;
use warchest::rng::{cumulative, Rng};
use crate::search::{judge, think, Choice, Oracle, Query, Search};
use warchest::state::State;

#[derive(Clone, Copy, serde::Deserialize)]
pub struct Human {
    pub temperature: f64,
    pub epsilon: f64,
}

#[derive(Clone)]
pub enum Opponent {
    Counted,
    Human(Human, Arc<dyn Oracle>),
}

impl Opponent {
    fn likelihood(&self, ctx: &Ctx, worlds: &[(State, Vec<Action>)]) -> Vec<f64> {
        let Opponent::Human(h, net) = self else {
            return ways(worlds);
        };
        let mut q = Query::default();
        for (s, acts) in worlds {
            q.push(s, ctx, acts);
        }
        let prior = net.eval(&q).prior;
        (0..q.len())
            .flat_map(|i| {
                let tempered: Vec<f64> = prior[q.actions(i)].iter().map(|&p| (p as f64).powf(1.0 / h.temperature)).collect();
                let (total, n) = (tempered.iter().sum::<f64>(), tempered.len() as f64);
                tempered.into_iter().map(move |p| (1.0 - h.epsilon) * p / total + h.epsilon / n)
            })
            .collect()
    }
}

pub enum Brain {
    Random,
    Greedy,
    Search { search: Search, oracle: Arc<dyn Oracle> },
}

impl Brain {
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
    opponent: Opponent,
    rng: Rng,
}

impl Session {
    pub fn new(draft: &Draft, seat: u8, seed: u64, opponent: Opponent) -> Result<Session, String> {
        let s = draft.state()?;
        Ok(Session {
            ctx: Ctx::new(&s),
            opp: Belief::new(&s, 1 - seat),
            opponent,
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

    fn watched(&mut self, player: u8, key: u32) -> Result<(), String> {
        if !self.to_move(player) {
            return Err(format!("player {} is not to act", player));
        }
        let (ctx, opponent) = (&self.ctx, &self.opponent);
        self.s = self
            .opp
            .update(&self.s, |a| obs_key(a) == key, |worlds| opponent.likelihood(ctx, worlds))
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

    pub fn judge(&mut self, search: Search, oracle: &dyn Oracle, played: Action) -> Result<(f32, f32), String> {
        let i = self.s.legal_actions().iter().position(|&a| a == played).ok_or("the played move is not legal here")?;
        Ok(judge(oracle, search, &self.s, &self.ctx, &self.opp, &mut self.rng, i))
    }

    pub fn belief(&self, truth: &State) -> f64 {
        self.opp.weight(truth)
    }

    pub fn query(&self, acts: &[Action], q: &mut Query) {
        q.push(&self.s, &self.ctx, acts);
    }
}

pub fn answer(ask: Ask, sessions: &Mutex<HashMap<u32, Session>>, brain: &Brain, opponent: &Opponent) -> Result<(Vec<u32>, Option<f32>), String> {
    let mut session = match ask.start {
        Some(start) => Session::new(&start.draft, start.seat, start.seed, opponent.clone())?,
        None => sessions.lock().unwrap().remove(&ask.id).ok_or_else(|| format!("game {} was never started", ask.id))?,
    };
    for obs in &ask.obs {
        session.observe(obs)?;
    }
    let (acts, choice) = session.think(brain)?;
    let value = (acts.len() > 1).then_some(choice.value);
    session.play(acts[choice.pick]);
    sessions.lock().unwrap().insert(ask.id, session);
    let mut ranked: Vec<usize> = (0..acts.len()).collect();
    ranked.sort_by(|&a, &b| (a != choice.pick).cmp(&(b != choice.pick)).then(choice.probs[b].total_cmp(&choice.probs[a])));
    Ok((ranked.into_iter().map(|i| acts[i].encode()).collect(), value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use warchest::arena::{Done, Table};

    fn draft() -> Draft {
        Draft {
            white: vec![17, 12, 4, 9],
            black: vec![1, 3, 8, 16],
            first: 0,
        }
    }

    #[test]
    fn a_human_reading_of_a_flat_prior_is_uniform_in_every_world() {
        let s = draft().state().unwrap();
        let mut w = s;
        while w.is_chance() {
            w.apply_inplace(w.legal_actions()[0]);
        }
        let worlds = vec![(w, w.legal_actions()); 2];
        let human = Opponent::Human(Human { temperature: 3.0, epsilon: 0.1 }, Arc::new(crate::search::Flat));
        let n = worlds[0].1.len() as f64;
        assert!(human.likelihood(&Ctx::new(&s), &worlds).iter().all(|&l| (l - 1.0 / n).abs() < 1e-6));
    }

    #[test]
    fn a_seat_is_told_nothing_private_and_follows_the_game() {
        let brain = Brain::Random;
        let mut table = Table::new();
        table.start(1, &draft(), [0, 1], 8191).unwrap();
        let mut seats = [
            Session::new(&draft(), 0, 1, Opponent::Counted).unwrap(),
            Session::new(&draft(), 1, 2, Opponent::Counted).unwrap(),
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
