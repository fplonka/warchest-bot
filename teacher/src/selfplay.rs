use std::sync::Arc;
use std::thread::JoinHandle;

use parking_lot::{Condvar, Mutex};
use rayon::prelude::*;

use warchest::actions::{Action, N_KINDS, N_PLAYS};
use warchest::arena::{dealt, Obs};
use engine::bot::{Brain, Session};
use warchest::actions::obs_key;
use warchest::rng::Rng;
use engine::search::{Answer, Oracle, Query, Search};
use warchest::rules::resolve_chance;
use warchest::state::{BLACK, WHITE};

pub const HORIZONS: [f32; 4] = [f32::INFINITY, 6.0, 16.0, 50.0];

fn targets(seen: &[(usize, f32, bool)], white: f32) -> Vec<[f32; 4]> {
    let mut t = [white; 4];
    let mut z: Vec<[f32; 4]> = seen
        .iter()
        .rev()
        .filter_map(|&(p, v, recorded)| {
            let sign = if p == WHITE as usize { 1.0 } else { -1.0 };
            for (t, h) in t.iter_mut().zip(HORIZONS) {
                *t += (sign * v - *t) / h;
            }
            recorded.then(|| t.map(|x| sign * x))
        })
        .collect();
    z.reverse();
    z
}

pub struct KindCounts(pub [[usize; 2]; N_KINDS]);

impl Default for KindCounts {
    fn default() -> Self {
        KindCounts([[0; 2]; N_KINDS])
    }
}

#[derive(Default)]
pub struct Data {
    pub q: Query,
    pub target: Vec<f32>,
    pub z: Vec<[f32; 4]>,
    pub games: usize,
    pub decisions: usize,
    pub wins: [usize; 2],
    pub draws: usize,
    pub timeouts: usize,
    pub plays: [usize; N_PLAYS],
    pub kinds: KindCounts,
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
        for (a, b) in self.kinds.0.iter_mut().zip(o.kinds.0) {
            a[0] += b[0];
            a[1] += b[1];
        }
    }

    fn count(&mut self, acts: &[Action], chosen: Action) {
        self.decisions += 1;
        if let Some(n) = self.plays.get_mut(chosen.play() as usize) {
            *n += 1;
        }
        let offered = acts.iter().fold(0u64, |m, a| m | 1 << a.kind());
        for (k, counts) in self.kinds.0.iter_mut().enumerate() {
            counts[1] += (offered >> k & 1) as usize;
        }
        self.kinds.0[chosen.kind()][0] += 1;
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
    let (mut d, mut seen) = (Data::default(), Vec::new());
    loop {
        while s.is_chance() {
            let p = s.to_act();
            let code = resolve_chance(&mut s, p, rng).encode();
            for (seat, sess) in seats.iter_mut().enumerate() {
                let obs = Obs::Draw { player: p, code: (seat == p as usize).then_some(code) };
                sess.observe(&obs).expect("a seat follows every draw");
            }
        }
        if warchest::finished(&s) {
            break;
        }
        if !open() {
            return None;
        }
        let p = s.to_act() as usize;
        let full = rng.unit_f64() < playout.p_full;
        let brain = if full { &playout.full } else { &playout.fast };
        let (acts, c) = seats[p].think(brain).expect("the seat to act can think");
        if acts.len() > 1 {
            if full {
                seats[p].query(&acts, &mut d.q);
                d.target.extend_from_slice(&c.probs);
            }
            seen.push((p, c.value, full));
        }
        let a = acts[c.pick];
        seats[p].play(a);
        d.count(&acts, a);
        s.apply_inplace(a);
        let obs = Obs::Act { player: p as u8, key: obs_key(&a) };
        seats[1 - p].observe(&obs).expect("a seat follows every move");
    }
    d.z = targets(&seen, warchest::score(&s, WHITE as usize));
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

struct Sampled {
    oracle: Arc<dyn Oracle>,
    keep: u64,
    kept: Mutex<Query>,
}

impl Oracle for Sampled {
    fn eval(&self, q: &Query) -> Answer {
        let picked: Vec<usize> = (0..q.len()).filter(|&i| digest(q.row(i).0) < self.keep).collect();
        if !picked.is_empty() {
            let mut kept = self.kept.lock();
            picked.into_iter().for_each(|i| kept.push_row(q.row(i)));
        }
        self.oracle.eval(q)
    }
}

fn digest(row: &[u8]) -> u64 {
    row.chunks(8).fold(0x9E37_79B9_7F4A_7C15, |h, c| {
        let x = c.iter().rev().fold(0u64, |x, &b| x << 8 | b as u64);
        (h ^ x).wrapping_mul(0xFF51_AFD7_ED55_8CCD).rotate_left(29)
    })
}

pub fn leaves(oracle: Arc<dyn Oracle>, sims: u32, keep: f64, games: usize, seed: u64) -> Query {
    let sampled = Arc::new(Sampled { oracle, keep: (keep * u64::MAX as f64) as u64, kept: Mutex::new(Query::default()) });
    let full = Brain::Search { search: Search::Full(sims), oracle: sampled.clone() };
    let playout = Playout { full, fast: Brain::Greedy, p_full: 1.0 };
    (0..games).into_par_iter().for_each(|i| {
        play(&playout, &mut Rng::new(worker_seed(seed, i)), || true);
    });
    let kept = std::mem::take(&mut *sampled.kept.lock());
    kept
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

fn worker_seed(seed: u64, i: usize) -> u64 {
    seed.wrapping_mul(0x9E3779B97F4A7C15) ^ (i as u64).wrapping_mul(0xD1B54A32D192ED03)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::pbs::ROW_BYTES;
    use engine::search::{Flat, Search, ACT_BYTES};

    fn searcher(p_full: f64) -> Playout {
        let search = |search| Brain::Search { search, oracle: Arc::new(Flat) };
        Playout { full: search(Search::Full(32)), fast: search(Search::Fast(16)), p_full }
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
    fn a_searching_game_yields_one_taught_row_per_real_choice() {
        let d = play(&searcher(1.0), &mut Rng::new(3), || true).expect("an unstopped game ends");
        check(&d);
        assert_eq!(d.wins[0] + d.wins[1] + d.draws + d.timeouts, 1);
        assert!(d.z.as_flattened().iter().all(|z| z.abs() <= 1.0));
    }

    #[test]
    fn short_term_targets_average_the_coming_root_values_and_leave_the_rest_to_the_result() {
        let z = targets(&[(0, 0.5, true), (1, 0.2, true), (0, -0.1, true)], 1.0);
        let six = [0.622_685, -0.647_222, 0.816_667];
        for (i, row) in z.iter().enumerate() {
            let sign = if i == 1 { -1.0 } else { 1.0 };
            assert_eq!(row[0], sign, "row {i} keeps the result");
            assert!((row[1] - six[i]).abs() < 1e-5, "row {i}: {} is not {}", row[1], six[i]);
        }
        let fifty = 0.02 * 0.5 - 0.02 * 0.98 * 0.2 - 0.02 * 0.98 * 0.98 * 0.1 + 0.98f32.powi(3);
        assert!((z[0][3] - fifty).abs() < 1e-5, "{} is not {fifty}", z[0][3]);
        assert_eq!(targets(&[(0, 0.5, true), (1, 0.2, false), (0, -0.1, true)], 1.0), [z[0], z[2]]);
    }

    #[test]
    fn fast_searches_record_no_rows() {
        let mut d = Data::default();
        for seed in 0..4 {
            d.merge(play(&searcher(0.5), &mut Rng::new(seed), || true).expect("an unstopped game ends"));
        }
        check(&d);
        let frac = d.q.len() as f64 / d.decisions as f64;
        assert!((0.25..0.6).contains(&frac), "{frac} of decisions are recorded");
    }

    #[test]
    fn greedy_warm_games_teach_every_row() {
        let d = warm(4, 11);
        check(&d);
        assert_eq!(d.kinds.0.iter().map(|[chosen, _]| chosen).sum::<usize>(), d.decisions);
        assert!(d.kinds.0.iter().all(|[chosen, offered]| chosen <= offered));
    }

    #[test]
    fn a_farm_of_searching_games_collects_rows() {
        let farm = Farm::new(5, 4, Arc::new(searcher(0.5)), 200);
        let d = farm.collect().expect("no game thread dies");
        drop(farm);
        check(&d);
    }
}
