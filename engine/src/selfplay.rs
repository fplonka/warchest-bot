use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use parking_lot::Mutex;
use rayon::prelude::*;

use crate::actions::{Action, N_PLAYS};
use crate::arena::{Draft, Obs};
use crate::board::NONE;
use crate::bot::{Brain, Session};
use crate::pbs::obs_key;
use crate::rng::Rng;
use crate::search::Query;
use crate::state::{Cont, State, BLACK, WHITE};

pub const DRAFT_POOL: [[u16; 2]; 16] = [
    [1, 1], [2, 52], [3, 3], [4, 4], [5, 5], [6, 53], [7, 7], [8, 8],
    [9, 9], [10, 10], [11, 11], [12, 12], [16, 16], [17, 17], [18, 54], [19, 19],
];

pub fn deal(rng: &mut Rng) -> [u16; 16] {
    let mut pool = DRAFT_POOL.map(|slot| slot[(rng.next_u64() & 1) as usize]);
    for i in (1..pool.len()).rev() {
        pool.swap(i, rng.below(i + 1));
    }
    pool
}

fn dealt(rng: &mut Rng) -> Draft {
    let first = (rng.next_u64() & 1) as u8;
    let pool = deal(rng);
    Draft { white: pool[..4].to_vec(), black: pool[4..8].to_vec(), first }
}

pub fn game(rng: &mut Rng) -> State {
    dealt(rng).state().expect("a dealt draft is legal")
}

#[derive(Default)]
pub struct Data {
    pub q: Query,
    pub target: Vec<f32>,
    pub z: Vec<f32>,
    pub games: usize,
    pub decisions: usize,
    pub wins: [usize; 2],
    pub draws: usize,
    pub timeouts: usize,
    pub plays: [usize; N_PLAYS],
}

impl Data {
    pub fn merge(&mut self, o: Data) {
        self.q.append(&o.q);
        self.target.extend(o.target);
        self.z.extend(o.z);
        self.games += o.games;
        self.decisions += o.decisions;
        self.draws += o.draws;
        self.timeouts += o.timeouts;
        for p in 0..2 {
            self.wins[p] += o.wins[p];
        }
        for (a, b) in self.plays.iter_mut().zip(o.plays) {
            *a += b;
        }
    }

    fn count(&mut self, chosen: Action) {
        self.decisions += 1;
        if let Some(n) = self.plays.get_mut(chosen.play() as usize) {
            *n += 1;
        }
    }
}

pub fn play(brain: &Brain, rng: &mut Rng, stop: &AtomicBool) -> Option<Data> {
    let draft = dealt(rng);
    let mut s = draft.state().expect("a dealt draft is legal");
    let mut seats = [WHITE, BLACK].map(|p| Session::new(&draft, p, rng.next_u64()).expect("a dealt draft is legal"));
    let (mut d, mut actors) = (Data::default(), Vec::new());
    loop {
        while s.is_chance() {
            let p = s.to_act();
            let code = resolve_chance(&mut s, p, rng).encode();
            for (seat, sess) in seats.iter_mut().enumerate() {
                let obs = Obs::Draw { player: p, code: (seat == p as usize).then_some(code) };
                sess.observe(&obs, brain).expect("a seat follows every draw");
            }
        }
        if crate::finished(&s) {
            break;
        }
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        let p = s.to_act() as usize;
        let (acts, probs) = seats[p].think(brain).expect("the seat to act can think");
        if acts.len() > 1 {
            seats[p].query(&acts, &mut d.q);
            d.target.extend_from_slice(&probs);
            actors.push(p);
        }
        let a = seats[p].play(brain, &acts, &probs);
        d.count(a);
        s.apply_inplace(a);
        let obs = Obs::Act { player: p as u8, key: obs_key(&a) };
        seats[1 - p].observe(&obs, brain).expect("a seat follows every move");
    }
    d.z = actors.iter().map(|&p| crate::score(&s, p)).collect();
    d.games = 1;
    match s.winner() {
        Some(w) => d.wins[w as usize] += 1,
        None if s.is_terminal() => d.draws += 1,
        None => d.timeouts += 1,
    }
    Some(d)
}

pub fn warm(brain: &Brain, games: usize, seed: u64) -> Data {
    let never = AtomicBool::new(false);
    (0..games)
        .into_par_iter()
        .map(|i| play(brain, &mut Rng::new(worker_seed(seed, i)), &never).expect("a warm game runs to its end"))
        .reduce(Data::default, |mut a, b| {
            a.merge(b);
            a
        })
}

pub struct Farm {
    stop: Arc<AtomicBool>,
    data: Arc<Mutex<Data>>,
    threads: Vec<JoinHandle<()>>,
}

impl Farm {
    pub fn new(seed: u64, games: usize, brain: Arc<Brain>) -> Farm {
        let stop = Arc::new(AtomicBool::new(false));
        let data = Arc::new(Mutex::new(Data::default()));
        let threads = (0..games)
            .map(|i| {
                let (brain, stop, data) = (Arc::clone(&brain), Arc::clone(&stop), Arc::clone(&data));
                std::thread::Builder::new()
                    .name(format!("game-{i}"))
                    .spawn(move || {
                        let mut rng = Rng::new(worker_seed(seed, i));
                        while let Some(d) = play(&brain, &mut rng, &stop) {
                            data.lock().merge(d);
                        }
                    })
                    .expect("spawn a game thread")
            })
            .collect();
        Farm { stop, data, threads }
    }

    pub fn rows(&self) -> usize {
        self.data.lock().q.len()
    }

    pub fn take(&self) -> Data {
        std::mem::take(&mut *self.data.lock())
    }

    pub fn alive(&self) -> bool {
        self.threads.iter().all(|t| !t.is_finished())
    }
}

impl Drop for Farm {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

pub(crate) fn resolve_chance(s: &mut State, player: u8, rng: &mut Rng) -> Action {
    debug_assert!(matches!(s.pending(), Cont::Draw { .. } | Cont::WarriorPriestDraw { .. }));
    let pool = s.draw_pool(player);
    let total = pool.iter().map(|&c| c as usize).sum::<usize>();
    let mut k = rng.below(total.max(1));
    let unit = pool.iter().position(|&c| match k.checked_sub(c as usize) {
        Some(rest) => {
            k = rest;
            false
        }
        None => true,
    });
    let drawn = Action::DrawCoin { unit: unit.map_or(NONE, |u| u as u8) };
    s.apply_inplace(drawn);
    drawn
}

fn worker_seed(seed: u64, i: usize) -> u64 {
    seed.wrapping_mul(0x9E3779B97F4A7C15) ^ (i as u64).wrapping_mul(0xD1B54A32D192ED03)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bot::Mind;
    use crate::pbs::ROW_BYTES;
    use crate::search::{uniform, Answer, Oracle, Search, ACT_BYTES};

    struct Flat;

    impl Oracle for Flat {
        fn eval(&self, q: Query) -> Answer {
            Answer { value: vec![0.0; q.len()], prior: uniform(&q) }
        }
    }

    fn searcher() -> Brain {
        let search = Search { sims: 24, batch: 8, puct: 1.5, explore: 0.25 };
        Brain { mind: Mind::Search { search, oracle: Arc::new(Flat) }, temp: 0.5, cut: 0.0 }
    }

    fn check(d: &Data) {
        assert!(d.games > 0 && d.q.len() > 20, "{} rows over {} games", d.q.len(), d.games);
        assert_eq!(d.q.rows.len(), d.q.len() * ROW_BYTES);
        assert_eq!(d.q.desc.len(), d.target.len() * ACT_BYTES);
        assert_eq!(d.z.len(), d.q.len());
        for i in 0..d.q.len() {
            let total: f32 = d.target[d.q.actions(i)].iter().sum();
            assert!((total - 1.0).abs() < 1e-4, "row {i} target sums to {total}");
        }
    }

    #[test]
    fn a_searching_game_yields_one_row_per_real_choice() {
        let d = play(&searcher(), &mut Rng::new(3), &AtomicBool::new(false)).expect("an unstopped game ends");
        check(&d);
        assert_eq!(d.wins[0] + d.wins[1] + d.draws + d.timeouts, 1);
        assert!(d.z.iter().all(|z| z.abs() <= 1.0));
    }

    #[test]
    fn greedy_warm_games_label_every_row() {
        check(&warm(&Brain { mind: Mind::Greedy, temp: 2.0, cut: 0.0 }, 4, 11));
    }

    #[test]
    fn a_farm_of_searching_games_collects_rows() {
        let farm = Farm::new(5, 4, Arc::new(searcher()));
        while farm.rows() < 200 {
            assert!(farm.alive(), "a game thread died");
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let d = farm.take();
        drop(farm);
        check(&d);
    }
}
