use std::time::Instant;

use warchest::pbs::{Belief, Ctx};
use warchest::rng::Rng;
use warchest::search::{think, Flat, Search};
use warchest::State;

fn roots(rng: &mut Rng) -> Vec<State> {
    let mut roots = Vec::new();
    while roots.len() < 200 {
        let (mut s, mut k) = (warchest::selfplay::game(rng), 0);
        while !warchest::finished(&s) {
            let acts = s.legal_actions();
            if !s.is_chance() && acts.len() > 1 {
                k += 1;
                if k % 7 == 0 {
                    roots.push(s);
                }
            }
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
    }
    roots
}

fn main() {
    let roots = roots(&mut Rng::new(5));
    for sims in [64, 512, 4096] {
        let ns = (0..3)
            .map(|_| {
                let (mut rng, t) = (Rng::new(9), Instant::now());
                for s in &roots {
                    std::hint::black_box(think(&Flat, Search::Full(sims), s, &Ctx::new(s), &Belief::new(s, 1 - s.to_act()), &mut rng));
                }
                t.elapsed().as_nanos() as f64 / (roots.len() * sims as usize) as f64
            })
            .fold(f64::INFINITY, f64::min);
        println!("sims={sims}: {ns:.0} ns/sim");
    }
}
