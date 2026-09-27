use crate::actions::Action;
use crate::board::{board, NONE, N_HEXES, N_LOCATIONS};
use crate::state::{State, Z_ELIM};

pub fn eval_static(s: &State, p: u8) -> f32 {
    if s.is_terminal() {
        return s.utility(p as usize) * 1e6;
    }
    let b = board();
    let o = 1 - p;
    let mut sc = 12.0 * (s.markers_on_board(p) as f32 - s.markers_on_board(o) as f32);
    let (mut coins_p, mut coins_o) = (0.0f32, 0.0f32);
    for h in 0..N_HEXES {
        match s.hex_owner[h] {
            x if x == p => coins_p += s.hex_height[h] as f32,
            x if x == o => coins_o += s.hex_height[h] as f32,
            _ => {}
        }
    }
    sc += 1.5 * (coins_p - coins_o);
    let elim = |q: u8| -> f32 { s.zones[q as usize][Z_ELIM].iter().map(|&x| x as f32).sum() };
    sc += elim(o) - elim(p);
    let (mut cover_p, mut cover_o) = (0.0f32, 0.0f32);
    for li in 0..N_LOCATIONS {
        let l = b.location_hexes[li] as usize;
        let (mut bp, mut bo) = (7.0f32, 7.0f32);
        for h in 0..N_HEXES {
            let owner = s.hex_owner[h];
            if owner == NONE {
                continue;
            }
            let d = b.dist[h][l] as f32;
            if owner == p {
                bp = bp.min(d);
            } else {
                bo = bo.min(d);
            }
        }
        if s.loc_marker[l] != p {
            cover_p += bp;
        }
        if s.loc_marker[l] != o {
            cover_o += bo;
        }
    }
    sc += 0.5 * (cover_o - cover_p);
    sc
}

pub fn greedy(s: &State, acts: &[Action]) -> Vec<f32> {
    let score: Vec<f32> = acts
        .iter()
        .map(|&a| eval_static(&s.apply(a), s.to_act()))
        .collect();
    let best = score.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = score.iter().map(|x| (x - best).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.into_iter().map(|x| x / sum).collect()
}
