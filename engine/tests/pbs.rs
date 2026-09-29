use std::collections::HashMap;

use warchest::pbs::*;
use warchest::rng::Rng;
use warchest::selfplay::game;
use warchest::state::State;
use warchest::units::{write_card_features, CARD_FEATS};

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
        let mut s = game(&mut Rng::new(g + 1));
        let ctx = Ctx::new(&s);
        for _ in 0..150 {
            if s.is_terminal() {
                break;
            }
            for p in 0..2u8 {
                let mut coins = unseen(&s, p);
                let (mut a, mut b) = (s, s);
                deal(&mut a, p, &coins);
                coins.reverse();
                deal(&mut b, p, &coins);
                let same = features(&a, &ctx) == features(&b, &ctx);
                if p != s.to_act() {
                    assert!(same, "features changed when only the waiting player's hidden coins changed");
                    hidden += 1;
                } else if a != b {
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
        let mut s = game(&mut rng);
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

