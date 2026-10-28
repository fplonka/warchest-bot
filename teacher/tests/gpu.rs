#![cfg(feature = "gpu")]

use std::sync::Arc;

use engine::bot::Brain;
use teacher::cuda::Gpu;
use teacher::net::Net;
use engine::pbs::Ctx;
use warchest::rng::Rng;
use engine::search::{Oracle, Query, Search};
use teacher::selfplay::{Farm, Playout};
use warchest::arena::game;

fn positions(games: u64) -> Query {
    let mut q = Query::default();
    for g in 0..games {
        let mut rng = Rng::new(g + 1);
        let mut s = game(&mut rng);
        let ctx = Ctx::new(&s);
        for _ in 0..120 {
            if s.is_terminal() {
                break;
            }
            let acts = s.legal_actions();
            if !s.is_chance() {
                q.push(&s, &ctx, &acts);
            }
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
    }
    q
}

#[test]
fn priors_are_distributions_and_values_are_bounded() {
    let q = positions(40);
    let a = Gpu::new(&[0], &Net::random(7)).unwrap().eval(&q);
    assert_eq!(a.value.len(), q.len());
    assert_eq!(a.prior.len(), q.off[q.len()] as usize);
    assert!(a.value.iter().all(|v| v.is_finite() && v.abs() <= 1.0));
    for i in 0..q.len() {
        let total: f32 = a.prior[q.actions(i)].iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "row {i} priors sum to {total}");
    }
    let spread = a.value.iter().copied().fold(f32::MIN, f32::max) - a.value.iter().copied().fold(f32::MAX, f32::min);
    assert!(spread > 1e-3, "every position got the same value");
}

#[test]
fn a_position_evaluates_the_same_alone_or_in_a_crowd() {
    let q = positions(60);
    let net = Net::random(11);
    let crowd = Gpu::new(&[0], &net).unwrap().eval(&q);
    let gpu = Gpu::new(&[0], &net).unwrap();
    for i in (0..q.len()).step_by(97) {
        let mut one = Query::default();
        one.push_row(q.row(i));
        let r = q.actions(i);
        let alone = gpu.eval(&one);
        assert!((alone.value[0] - crowd.value[i]).abs() < 1e-4, "row {i} value");
        for (x, y) in alone.prior.iter().zip(&crowd.prior[r]) {
            assert!((x - y).abs() < 1e-4, "row {i} prior {x} vs {y}");
        }
    }
}

#[test]
fn a_farm_searches_on_the_card() {
    let gpu = Arc::new(Gpu::new(&[0], &Net::random(3)).unwrap());
    let search = |search| Brain::Search { search, oracle: gpu.clone() };
    let playout = Playout { full: search(Search::Full(128)), fast: search(Search::Fast(16)), p_full: 0.5 };
    let farm = Farm::new(1, 64, Arc::new(playout), 2000);
    let t0 = std::time::Instant::now();
    let d = farm.collect().expect("no game thread dies");
    let seconds = t0.elapsed().as_secs_f64();
    let rows = gpu.stats[0].rows.load(std::sync::atomic::Ordering::Relaxed);
    eprintln!("{} games, {} decisions, {rows} evals in {seconds:.1}s: {:.0} evals/s", d.games, d.decisions, rows as f64 / seconds);
    drop(farm);
}
