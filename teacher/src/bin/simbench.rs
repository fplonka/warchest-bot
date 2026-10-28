use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use engine::bot::Brain;
use warchest::rng::Rng;
use engine::search::{Flat, Search};
use teacher::selfplay::{play, Playout};

fn main() {
    let arg = |i: usize, default: &str| std::env::args().nth(i).unwrap_or(default.into());
    let (games, sims, p_full): (u64, u32, f64) = (
        arg(1, "20").parse().expect("games is a count"),
        arg(2, "32").parse().expect("sims is a count"),
        arg(3, "1").parse().expect("p_full is a probability"),
    );
    let search = |search| Brain::Search { search, oracle: Arc::new(Flat) };
    let playout = Playout { full: search(Search::Full(sims)), fast: search(Search::Fast(16)), p_full };
    let (mut h, mut decisions, mut rows, t) = (DefaultHasher::new(), 0, 0, Instant::now());
    for seed in 0..games {
        let d = play(&playout, &mut Rng::new(seed), || true).expect("an unstopped game ends");
        (&d.q.rows, &d.q.desc, &d.q.off).hash(&mut h);
        d.target.iter().chain(d.z.as_flattened()).for_each(|x| x.to_bits().hash(&mut h));
        (decisions, rows) = (decisions + d.decisions, rows + d.q.len());
    }
    let s = t.elapsed().as_secs_f64();
    println!(
        "games={games} sims={sims} p_full={p_full}: {decisions} decisions, {rows} rows, {:.0} decisions/s, hash {:016x}",
        decisions as f64 / s,
        h.finish()
    );
}
