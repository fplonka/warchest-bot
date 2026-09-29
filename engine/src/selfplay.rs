use std::sync::Arc;
use std::thread::JoinHandle;

use parking_lot::{Condvar, Mutex};
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
    rng.shuffle(&mut pool);
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

pub struct Playout {
    pub full: Brain,
    pub fast: Brain,
    pub p_full: f64,
}

pub fn play(playout: &Playout, rng: &mut Rng, open: impl Fn() -> bool) -> Option<Data> {
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
                sess.observe(&obs).expect("a seat follows every draw");
            }
        }
        if crate::finished(&s) {
            break;
        }
        if !open() {
            return None;
        }
        let p = s.to_act() as usize;
        let full = rng.unit_f64() < playout.p_full;
        let brain = if full { &playout.full } else { &playout.fast };
        let (acts, probs, pick) = seats[p].think(brain).expect("the seat to act can think");
        if acts.len() > 1 {
            let taught = if full { &acts[..] } else { &[] };
            seats[p].query(taught, &mut d.q);
            d.target.extend_from_slice(&probs[..taught.len()]);
            actors.push(p);
        }
        let a = acts[pick];
        seats[p].play(a);
        d.count(a);
        s.apply_inplace(a);
        let obs = Obs::Act { player: p as u8, key: obs_key(&a) };
        seats[1 - p].observe(&obs).expect("a seat follows every move");
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

pub fn warm(games: usize, seed: u64) -> Data {
    let greedy = Playout { full: Brain::Greedy, fast: Brain::Greedy, p_full: 1.0 };
    (0..games)
        .into_par_iter()
        .map(|i| play(&greedy, &mut Rng::new(worker_seed(seed, i)), || true).expect("a warm game runs to its end"))
        .reduce(Data::default, |mut a, b| {
            a.merge(b);
            a
        })
}

pub struct Farm {
    pool: Arc<(Mutex<(Data, bool)>, Condvar)>,
    limit: usize,
    threads: Vec<JoinHandle<()>>,
}

impl Farm {
    pub fn new(seed: u64, games: usize, playout: Arc<Playout>, limit: usize) -> Farm {
        let pool = Arc::new((Mutex::new((Data::default(), false)), Condvar::new()));
        let threads = (0..games)
            .map(|i| {
                let (playout, pool) = (Arc::clone(&playout), Arc::clone(&pool));
                std::thread::Builder::new()
                    .name(format!("game-{i}"))
                    .spawn(move || {
                        let mut rng = Rng::new(worker_seed(seed, i));
                        let open = || {
                            let mut held = pool.0.lock();
                            pool.1.wait_while(&mut held, |(d, stop)| d.q.len() >= limit && !*stop);
                            !held.1
                        };
                        while let Some(d) = play(&playout, &mut rng, open) {
                            pool.0.lock().0.merge(d);
                        }
                    })
                    .expect("spawn a game thread")
            })
            .collect();
        Farm { pool, limit, threads }
    }

    pub fn collect(&self) -> Option<Data> {
        while self.pool.0.lock().0.q.len() < self.limit {
            if self.threads.iter().any(|t| t.is_finished()) {
                return None;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let d = std::mem::take(&mut self.pool.0.lock().0);
        self.pool.1.notify_all();
        Some(d)
    }
}

impl Drop for Farm {
    fn drop(&mut self) {
        self.pool.0.lock().1 = true;
        self.pool.1.notify_all();
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
    use crate::pbs::ROW_BYTES;
    use crate::search::{Flat, Search, ACT_BYTES};

    fn searcher(p_full: f64) -> Playout {
        let search = |search| Brain::Search { search, oracle: Arc::new(Flat) };
        Playout { full: search(Search::Full(32)), fast: search(Search::Fast(16)), p_full }
    }

    fn taught(d: &Data) -> usize {
        assert!(d.games > 0 && d.q.len() > 20, "{} rows over {} games", d.q.len(), d.games);
        assert_eq!(d.q.rows.len(), d.q.len() * ROW_BYTES);
        assert_eq!(d.q.desc.len(), d.target.len() * ACT_BYTES);
        assert_eq!(d.z.len(), d.q.len());
        let mut taught = 0;
        for i in 0..d.q.len() {
            let total: f32 = d.target[d.q.actions(i)].iter().sum();
            assert!((total - 1.0).abs() < 1e-4 || d.q.actions(i).is_empty(), "row {i} target sums to {total}");
            taught += !d.q.actions(i).is_empty() as usize;
        }
        taught
    }

    #[test]
    fn a_searching_game_yields_one_taught_row_per_real_choice() {
        let d = play(&searcher(1.0), &mut Rng::new(3), || true).expect("an unstopped game ends");
        assert_eq!(taught(&d), d.q.len());
        assert_eq!(d.wins[0] + d.wins[1] + d.draws + d.timeouts, 1);
        assert!(d.z.iter().all(|z| z.abs() <= 1.0));
    }

    #[test]
    fn fast_searches_record_rows_without_a_policy_target() {
        let mut d = Data::default();
        for seed in 0..4 {
            d.merge(play(&searcher(0.5), &mut Rng::new(seed), || true).expect("an unstopped game ends"));
        }
        let frac = taught(&d) as f64 / d.q.len() as f64;
        assert!((0.4..0.6).contains(&frac), "{frac} of rows are taught");
    }

    #[test]
    fn greedy_warm_games_teach_every_row() {
        let d = warm(4, 11);
        assert_eq!(taught(&d), d.q.len());
    }

    #[test]
    fn a_farm_of_searching_games_collects_rows() {
        let farm = Farm::new(5, 4, Arc::new(searcher(0.5)), 200);
        let d = farm.collect().expect("no game thread dies");
        drop(farm);
        taught(&d);
    }
}
