use std::collections::HashMap;

use warchest::pbs::*;
use warchest::rng::Rng;
use warchest::search::uniform;
use warchest::selfplay::make_game;
use warchest::state::{Cont, State, Z_BAG, Z_FACEDOWN, Z_FACEUP};
use warchest::units::{write_card_features, CARD_FEATS};
use warchest::Action;

fn world(pubs: &State, ctx: &Ctx, c: &(Config, Config)) -> State {
    let mut w = *pubs;
    set_config(&mut w, 0, ctx, &c.0);
    set_config(&mut w, 1, ctx, &c.1);
    w
}

fn pick(c: &(Config, Config), p: u8) -> &Config {
    if p == 0 {
        &c.0
    } else {
        &c.1
    }
}

fn with(c: &(Config, Config), p: u8, n: Config) -> (Config, Config) {
    if p == 0 {
        (n, c.1)
    } else {
        (c.0, n)
    }
}

fn draw_weights(s: &State, p: u8, acts: &[Action]) -> Vec<f64> {
    let bag_total: u8 = s.zones[p as usize][Z_BAG].iter().sum();
    acts.iter()
        .map(|a| match a {
            Action::DrawCoin { unit } => {
                let u = *unit as usize;
                if bag_total > 0 {
                    s.zones[p as usize][Z_BAG][u] as f64
                } else {
                    (s.zones[p as usize][Z_FACEUP][u] + s.zones[p as usize][Z_FACEDOWN][u]) as f64
                }
            }
            _ => 0.0,
        })
        .collect()
}

fn features(s: &State, ctx: &Ctx) -> Vec<f32> {
    let (mut row, mut x) = ([0u8; ROW_BYTES], vec![0.0f32; PUBFEAT]);
    pack_row(s, ctx, &mut row);
    expand_row(&row, &mut x);
    x
}

#[test]
fn a_row_shows_the_actor_its_hand_and_hides_the_other_hand() {
    let (mut rng, mut hidden, mut shown) = (Rng::new(99), 0, 0);
    for g in 0..12u64 {
        let mut s = make_game(&mut Rng::new(g + 1), true);
        let ctx = Ctx::new(&s);
        for _ in 0..150 {
            if s.is_terminal() {
                break;
            }
            for p in 0..2u8 {
                let truth = true_config(&s, p, &ctx);
                let all = enumerate_configs(&reserve(&s, p, &ctx), truth.hand_size(), truth.fd_size(), truth.inflight.is_some());
                let (mut a, mut b) = (s, s);
                set_config(&mut a, p, &ctx, &all[0]);
                set_config(&mut b, p, &ctx, &all[all.len() - 1]);
                let same = features(&a, &ctx) == features(&b, &ctx);
                if p != s.to_act() {
                    assert!(same, "features changed when only the waiting player's hidden config changed");
                    hidden += 1;
                } else if all.len() > 1 {
                    assert!(!same, "the actor's own hidden config is missing from its features");
                    shown += 1;
                }
            }
            let acts = s.legal_actions();
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
    }
    assert!(hidden > 500 && shown > 100, "{hidden} hidden and {shown} shown positions");
}

#[test]
fn belief_tracker_matches_brute_force() {
    for seed in 0..6u64 {
        run_one_draft(seed, &[17, 12, 4, 9], &[1, 3, 8, 16]);
    }
    for seed in 0..4u64 {
        run_one_draft(seed + 100, &[18, 17, 12, 4], &[54, 1, 3, 8]);
    }
}

fn run_one_draft(seed: u64, white: &[u16], black: &[u16]) {
    let mut rng = Rng::new(seed + 1);
    let mut s = State::from_draft(white, black, warchest::state::WHITE);
    let ctx = Ctx::new(&s);
    let mut bel = [Belief::point(Config::default()), Belief::point(Config::default())];
    let mut worlds: HashMap<(Config, Config), f64> = HashMap::new();
    worlds.insert((Config::default(), Config::default()), 1.0);

    let mut steps = 0;
    while !s.is_terminal() && steps < 60 && worlds.len() < 20_000 {
        let p = s.to_act();
        if s.is_chance() {
            let mut next: HashMap<(Config, Config), f64> = HashMap::new();
            for (c, w) in worlds.iter() {
                let ws = world(&s, &ctx, c);
                let acts = ws.legal_actions();
                let wts = draw_weights(&ws, p, &acts);
                let tot: f64 = wts.iter().sum();
                assert!(tot > 0.0, "a world had nothing to draw");
                for (i, a) in acts.iter().enumerate() {
                    if wts[i] <= 0.0 {
                        continue;
                    }
                    let mut nx = ws;
                    nx.apply_inplace(*a);
                    let nc = with(c, p, true_config(&nx, p, &ctx));
                    *next.entry(nc).or_insert(0.0) += w * wts[i] / tot;
                }
            }
            worlds = next;

            let res = reserve(&s, p, &ctx);
            let fu = faceup_counts(&s, p, &ctx);
            let wp = matches!(s.pending(), Cont::WarriorPriestDraw { .. });
            bel[p as usize] = belief_after_draw(&bel[p as usize], &res, &fu, wp);

            let acts = s.legal_actions();
            let wts = draw_weights(&s, p, &acts);
            let ai = rng.weighted_index(&wts);
            s.apply_inplace(acts[ai]);
            compare(&worlds, &bel, seed, steps, "draw");
            steps += 1;
            continue;
        }

        let acts = s.legal_actions();
        let chosen = rng.below(acts.len());
        let obs = obs_key(&acts[chosen]);

        let mut next: HashMap<(Config, Config), f64> = HashMap::new();
        for (c, w) in worlds.iter() {
            let ws = world(&s, &ctx, c);
            let wacts = ws.legal_actions();
            for a in wacts.iter().filter(|a| obs_key(a) == obs) {
                let nx = ws.apply(*a);
                let nc = with(c, p, true_config(&nx, p, &ctx));
                *next.entry(nc).or_insert(0.0) += w / wacts.len() as f64;
            }
        }
        let tot: f64 = next.values().sum();
        assert!(tot > 0.0, "brute force lost every world at step {}", steps);
        for v in next.values_mut() {
            *v /= tot;
        }
        worlds = next;

        bel[p as usize] = posterior(&s, &ctx, p, &bel[p as usize], obs, uniform).0;

        s.apply_inplace(acts[chosen]);
        compare(&worlds, &bel, seed, steps, "decision");
        steps += 1;
    }
    assert!(
        steps > 8,
        "seed {} produced too short a trace ({} steps, {} worlds)",
        seed,
        steps,
        worlds.len()
    );
    eprintln!(
        "seed {}: verified {} steps against {} exhaustively enumerated worlds",
        seed,
        steps,
        worlds.len()
    );
}

fn compare(worlds: &HashMap<(Config, Config), f64>, bel: &[Belief; 2], seed: u64, step: usize, what: &str) {
    for p in 0..2u8 {
        let mut exact: HashMap<Config, f64> = HashMap::new();
        for (c, w) in worlds.iter() {
            *exact.entry(*pick(c, p)).or_insert(0.0) += w;
        }
        let tot: f64 = exact.values().sum();
        for v in exact.values_mut() {
            *v /= tot;
        }
        let b = &bel[p as usize];
        for (c, w) in exact.iter() {
            if *w < 1e-9 {
                continue;
            }
            let got = b.cfg.iter().position(|x| x == c).map(|i| b.p[i] as f64).unwrap_or(0.0);
            assert!(
                (got - w).abs() < 1e-5,
                "seed {} step {} ({}) player {}: config {:?} exact {:.9} tracker {:.9}",
                seed,
                step,
                what,
                p,
                c,
                w,
                got
            );
        }
        for (i, c) in b.cfg.iter().enumerate() {
            if b.p[i] < 1e-9 {
                continue;
            }
            let w = exact.get(c).copied().unwrap_or(0.0);
            assert!(
                (b.p[i] as f64 - w).abs() < 1e-5,
                "seed {} step {} ({}) player {}: tracker has a phantom config {:?} at {:.9}",
                seed,
                step,
                what,
                p,
                c,
                b.p[i]
            );
        }
    }
}

#[test]
fn card_features_separate_every_draftable_unit() {
    let mut seen: HashMap<Vec<u32>, u8> = HashMap::new();
    for u in 0..warchest::units::N_UNITS as u8 {
        let mut f = vec![0.0f32; CARD_FEATS];
        write_card_features(u, &mut f);
        let key: Vec<u32> = f.iter().map(|x| x.to_bits()).collect();
        if let Some(prev) = seen.insert(key, u) {
            assert_eq!(
                prev, u,
                "units {} and {} share a card vector, so the describer cannot \
                 tell them apart",
                prev, u
            );
        }
    }
}

#[test]
fn a_mirrored_row_is_the_mirrored_state_packed() {
    let mut checked = 0usize;
    for seed in 0..40u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9) | 1);
        let mut s = make_game(&mut rng, true);
        let ctx = Ctx::new(&s);
        for _ in 0..120 {
            if s.is_terminal() {
                break;
            }
            if s.is_valued() {
                let (mut a, mut b, mut got) = ([0u8; ROW_BYTES], [0u8; ROW_BYTES], [0u8; ROW_BYTES]);
                pack_row(&s, &ctx, &mut a);
                pack_row(&s.mirror(), &ctx.mirrored(), &mut b);
                mirror_row(&a, &mut got);
                assert_eq!(got, b, "seed {seed}: mirror_row disagrees with State::mirror");
                let mut back = [0u8; ROW_BYTES];
                mirror_row(&got, &mut back);
                assert_eq!(back, a, "seed {seed}: mirror is not an involution");
                checked += 1;
            }
            let acts = s.legal_actions();
            s.apply_inplace(acts[rng.below(acts.len())]);
        }
    }
    assert!(checked > 500, "only {checked} rows mirrored");
}

