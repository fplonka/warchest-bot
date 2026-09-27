use warchest::board::{board, NONE, N_HEXES, N_LOCATIONS};
use warchest::rng::Rng;
use warchest::selfplay::{game, DRAFT_POOL};
use warchest::state::{Cont, State, BLACK, WHITE, Z_INFLIGHT};
use warchest::units::{def, N_UNITS, ROYAL_COIN};

fn totals(s: &State, p: u8) -> [u16; N_UNITS] {
    std::array::from_fn(|u| s.total_coins(p, u) as u16)
}

fn check_invariants(s: &State, init: &[[u16; N_UNITS]; 2]) {
    for p in 0..2u8 {
        for u in 0..N_UNITS {
            let got = s.total_coins(p, u) as u16;
            assert_eq!(
                got, init[p as usize][u],
                "coin conservation broke: player {} unit {} got {} want {}",
                p, u, got, init[p as usize][u]
            );
        }
    }
    let b = board();
    for h in 0..N_HEXES {
        if s.hex_type[h] != NONE {
            assert!(s.hex_owner[h] == WHITE || s.hex_owner[h] == BLACK);
            assert!(s.hex_height[h] >= 1, "deployed stack must be >= 1");
        } else {
            assert_eq!(s.hex_height[h], 0);
            assert_eq!(s.hex_owner[h], NONE);
        }
        if s.loc_marker[h] != NONE {
            assert!(b.is_location[h], "marker on a non-location hex");
        }
    }
    for p in 0..2u8 {
        let on = s.markers_on_board(p);
        assert!(on <= 6);
        assert_eq!(on + s.markers_hand[p as usize], 6, "marker count broke");
    }
    for p in 0..2u8 {
        assert!(
            s.hand_size(p) <= 3,
            "hand held {} coins: a WP draw pushed it past the cap",
            s.hand_size(p)
        );
        let owed = usize::from(matches!(
            *s.pending(),
            Cont::WarriorPriestPlay { player } if player == p
        )) + s
            .conts
            .iter()
            .filter(|c| matches!(c, Cont::WarriorPriestPlay { player } if *player == p))
            .count();
        let flight: usize = s.zones[p as usize][Z_INFLIGHT].iter().map(|&n| n as usize).sum();
        if s.is_terminal() {
            assert_eq!(flight, 0, "a terminal has spent the in-flight coin");
        } else {
            assert_eq!(flight, owed, "in-flight zone must match the tree");
        }
        assert!(flight <= 1, "at most one coin in flight");
    }
    if let Some(w) = s.winner() {
        assert_eq!(s.markers_on_board(w), 6, "winner must have 6 markers");
    }
    if s.is_terminal() && s.winner().is_none() {
        assert_eq!(s.utility(0), 0.0);
        assert_eq!(s.utility(1), 0.0);
    }
}

#[test]
fn invariants_random_playouts() {
    let n_games = 10_000;
    let cap = 1000u32;
    let mut rng = Rng::new(0xA11CE);
    let mut cap_hits = 0u32;
    let mut lengths_sum = 0u64;
    let mut winners = [0u32; 3];

    for _ in 0..n_games {
        let mut s = game(&mut rng);
        let init = [totals(&s, 0), totals(&s, 1)];
        check_invariants(&s, &init);

        let mut applies = 0u32;
        while !s.is_terminal() && applies < cap {
            let acts = s.legal_actions();
            assert!(
                !acts.is_empty(),
                "no legal actions at a non-terminal state: {}",
                s.pending_debug()
            );
            let a = acts[rng.below(acts.len())];
            s.apply_inplace(a);
            check_invariants(&s, &init);
            applies += 1;
        }
        lengths_sum += applies as u64;
        if s.is_terminal() {
            match s.winner() {
                Some(winner) => winners[winner as usize] += 1,
                None => winners[2] += 1,
            }
        } else {
            cap_hits += 1;
            winners[2] += 1;
        }
    }

    eprintln!(
        "invariants: {} games, avg len {:.1}, cap hits {}, winners W={} B={} none={}",
        n_games,
        lengths_sum as f64 / n_games as f64,
        cap_hits,
        winners[0],
        winners[1],
        winners[2],
    );
    if cap_hits > 0 {
        eprintln!("WARNING: {} playouts hit the 1000-action cap", cap_hits);
    }
}

#[test]
fn random_drafts_never_duplicate_a_unit_card() {
    for g in 0..5000u64 {
        let s = game(&mut Rng::new(g));
        for u in 0..N_UNITS {
            if u as u8 == ROYAL_COIN {
                continue;
            }
            let (w, b) = (s.total_coins(0, u), s.total_coins(1, u));
            assert!(
                w == 0 || b == 0,
                "game {g}: unit {u} drafted by both players ({w} and {b} coins)"
            );
            assert!(
                w + b <= def(u as u8).coins,
                "game {g}: unit {u} has {} coins in play, the card has {}",
                w + b,
                def(u as u8).coins
            );
        }
        let dealt = |id: u16| {
            let u = warchest::units::index_of_id(id).unwrap() as usize;
            s.total_coins(0, u) + s.total_coins(1, u) > 0
        };
        for [base, revised] in DRAFT_POOL.into_iter().filter(|slot| slot[0] != slot[1]) {
            assert!(
                !(dealt(base) && dealt(revised)),
                "game {g}: units {base} and {revised} are two versions of one card"
            );
        }
    }
}

#[test]
fn the_exported_board_tables_describe_the_board() {
    let bd = board();
    let nb = warchest::board::neighbour_gather();
    assert_eq!(nb.len(), N_HEXES * 6);
    for h in 0..N_HEXES {
        for d in 0..6 {
            let n = nb[h * 6 + d] as usize;
            assert!(n <= N_HEXES, "hex {h} dir {d}: {n} is not a hex or the pad");
            assert_eq!(
                n == N_HEXES,
                bd.neighbors[h][d] == NONE,
                "hex {h} dir {d}: padding must mean exactly 'no neighbour'"
            );
            if n < N_HEXES {
                assert!(
                    nb[n * 6..(n + 1) * 6].contains(&(h as u8)),
                    "hex {h} lists {n} as a neighbour but {n} does not list {h}"
                );
            }
        }
    }

    let loc = bd.location_hexes;
    assert_eq!(loc.len(), N_LOCATIONS);
    let mut marked: Vec<u8> = (0..N_HEXES as u8).filter(|&h| bd.is_location[h as usize]).collect();
    let mut exported = loc.to_vec();
    exported.sort_unstable();
    marked.sort_unstable();
    assert_eq!(exported, marked, "location_hexes must be the marked hexes");
}
