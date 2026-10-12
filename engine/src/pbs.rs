use std::collections::BTreeMap;

use crate::actions::{Action, N_KINDS};
use crate::board::{board, NONE, N_HEXES};
use crate::rng::Rng;
use crate::state::{State, CONT_CAP, PENDING_KINDS, Z_BAG, Z_ELIM, Z_FACEDOWN, Z_FACEUP, Z_HAND, Z_INFLIGHT, Z_SUPPLY};
use crate::units::{write_card_features, CARD_FEATS, ENSIGN, MARSHAL, N_UNITS, ROYAL_COIN};

pub const NSLOT: usize = 5;
const HIDDEN: [usize; 4] = [Z_INFLIGHT, Z_HAND, Z_FACEDOWN, Z_BAG];

#[derive(Clone, Copy, Default)]
pub struct Ctx {
    pub slots: [[u8; NSLOT]; 2],
    pub slot_of: [[i8; N_UNITS]; 2],
}

impl Ctx {
    pub fn new(s: &State) -> Ctx {
        let mut slots = [[0u8; NSLOT]; 2];
        let mut slot_of = [[-1i8; N_UNITS]; 2];
        for p in 0..2usize {
            let mut n = 0;
            for u in 0..N_UNITS {
                if s.total_coins(p as u8, u) > 0 {
                    assert!(n < NSLOT, "a player may own at most {} coin types", NSLOT);
                    slots[p][n] = u as u8;
                    slot_of[p][u] = n as i8;
                    n += 1;
                }
            }
            assert_eq!(n, NSLOT, "expected exactly {} coin types per player", NSLOT);
        }
        Ctx { slots, slot_of }
    }

    pub fn mirrored(self) -> Ctx {
        Ctx {
            slots: [self.slots[1], self.slots[0]],
            slot_of: [self.slot_of[1], self.slot_of[0]],
        }
    }
}

fn reserve(s: &State, p: u8, ctx: &Ctx) -> [u8; NSLOT] {
    ctx.slots[p as usize].map(|u| HIDDEN.iter().map(|&z| s.zones[p as usize][z][u as usize]).sum())
}

type Layout = [[u8; N_UNITS]; HIDDEN.len()];

fn layout(s: &State, p: u8) -> Layout {
    HIDDEN.map(|z| s.zones[p as usize][z])
}

fn put(s: &mut State, p: u8, k: &Layout) {
    for (z, zone) in HIDDEN.into_iter().zip(k) {
        s.zones[p as usize][z] = *zone;
    }
}

pub struct Belief {
    p: u8,
    layouts: Vec<Layout>,
    weights: Vec<f64>,
}

impl Belief {
    pub fn new(s: &State, p: u8) -> Belief {
        Belief { p, layouts: vec![layout(s, p)], weights: vec![1.0] }
    }

    pub fn deal(&self, s: &mut State, rng: &mut Rng) {
        put(s, self.p, &self.layouts[rng.weighted_index(&self.weights)]);
    }

    pub fn update(
        &mut self,
        s: &State,
        seen: impl Fn(&Action) -> bool,
        likelihood: impl FnOnce(&[(State, Vec<Action>)]) -> Vec<f64>,
    ) -> Option<State> {
        let (worlds, weights): (Vec<_>, Vec<f64>) = self
            .layouts
            .iter()
            .zip(&self.weights)
            .filter_map(|(k, &x)| {
                let mut w = *s;
                put(&mut w, self.p, k);
                let acts = w.legal_actions();
                acts.iter().any(&seen).then_some(((w, acts), x))
            })
            .unzip();
        let mut odds = likelihood(&worlds).into_iter();
        let (mut next, mut after) = (BTreeMap::new(), None);
        for ((w, acts), x) in worlds.iter().zip(weights) {
            for (a, l) in acts.iter().zip(odds.by_ref()) {
                if seen(a) {
                    let child = w.apply(*a);
                    *next.entry(layout(&child, self.p)).or_insert(0.0) += x * l;
                    after = Some(child);
                }
            }
        }
        let after = after?;
        let total: f64 = next.values().sum();
        (self.layouts, self.weights) = next.into_iter().map(|(k, x)| (k, x / total)).unzip();
        Some(after)
    }
}

pub fn ways(worlds: &[(State, Vec<Action>)]) -> Vec<f64> {
    worlds.iter().flat_map(|(s, acts)| acts.iter().map(|a| copies(s, a) as f64)).collect()
}

fn copies(s: &State, a: &Action) -> u8 {
    let p = s.to_act();
    match *a {
        Action::DrawCoin { unit } if unit != NONE => s.draw_pool(p)[unit as usize],
        _ => match action_coin(a, s) {
            NONE => 1,
            coin => s.zones[p as usize][s.pay_zone()][coin as usize],
        },
    }
}

pub(crate) fn action_coin(a: &Action, s: &State) -> u8 {
    use Action::*;
    match *a {
        Deploy { unit, .. } | Bolster { unit, .. } => unit,
        ClaimInitiative { coin } | Recruit { coin, .. } | Pass { coin } => coin,
        TacRoyalGuard { .. } => ROYAL_COIN,
        TacEnsign { .. } => ENSIGN,
        TacMarshal { .. } => MARSHAL,
        Move { from, .. }
        | Control { from }
        | Attack { from, .. }
        | TacArcher { from, .. }
        | TacCavalryMove { from, .. }
        | TacCrossbow { from, .. }
        | TacLancer { from, .. }
        | TacLightCav { from, .. } => s.hex_type[from as usize],
        _ => NONE,
    }
}

pub fn obs_key(a: &Action) -> u32 {
    let code = a.encode();
    match a {
        Action::Pass { .. } | Action::ClaimInitiative { .. } => code & 63,
        Action::Recruit { unit, .. } => (code & 63) | ((*unit as u32) << 12),
        _ => code,
    }
}

pub const MAX_COINS: f32 = 4.0 * 5.0 + 1.0;

pub const NTYPE: usize = 2 * NSLOT;

pub const HEX_FACTS: usize = 2 + 1 + 2 + 1 + CONT_CAP;
pub const HEX_CH: usize = HEX_FACTS + NTYPE;
pub const HEX_BLOCK: usize = N_HEXES * HEX_CH;
pub const PILE_COUNTS: usize = 7;
pub const PLAYER_SCALARS: usize = 6;
pub const GLOBAL_SCALARS: usize = 3 + CONT_CAP * PENDING_KINDS;

pub const OFF_PILES: usize = HEX_BLOCK;
pub const OFF_CARDS: usize = OFF_PILES + NTYPE * PILE_COUNTS;
pub const OFF_LOOSE: usize = OFF_CARDS + NTYPE * CARD_FEATS;
pub const LOOSE: usize = 2 * PLAYER_SCALARS + GLOBAL_SCALARS;

pub const ROW_IDS: usize = 0;
pub const ROW_HEX_OWNER: usize = ROW_IDS + 2 * NSLOT;
pub const ROW_HEX_SLOT: usize = ROW_HEX_OWNER + N_HEXES;
pub const ROW_HEX_HEIGHT: usize = ROW_HEX_SLOT + N_HEXES;
pub const ROW_HEX_MARKER: usize = ROW_HEX_HEIGHT + N_HEXES;
pub const ROW_PILES: usize = ROW_HEX_MARKER + N_HEXES;
pub const ROW_HAND_SIZE: usize = ROW_PILES + 2 * NSLOT * PILE_COUNTS;
pub const ROW_FD_SIZE: usize = ROW_HAND_SIZE + 2;
pub const ROW_BAG_SIZE: usize = ROW_FD_SIZE + 2;
pub const ROW_INITIATIVE: usize = ROW_BAG_SIZE + 2;
pub const ROW_INIT_MOVED: usize = ROW_INITIATIVE + 1;
pub const ROW_TO_ACT: usize = ROW_INIT_MOVED + 1;
pub const ROW_WP: usize = ROW_TO_ACT + 1;
pub const ROW_STACK_KIND: usize = ROW_WP + 1;
pub const ROW_STACK_OWED: usize = ROW_STACK_KIND + CONT_CAP;
pub const ROW_BYTES: usize = ROW_STACK_OWED + CONT_CAP * 8;

pub fn rules_table_hash() -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x1000_0000_01b3);
    };
    mix(N_KINDS as u64);
    let bd = board();
    for &l in bd.is_location.iter() {
        mix(l as u64);
    }
    for u in 0..N_UNITS {
        let mut f = [0.0f32; CARD_FEATS];
        write_card_features(u as u8, &mut f);
        for x in f {
            mix(x.to_bits() as u64);
        }
    }
    h
}

pub const PUBFEAT: usize = OFF_LOOSE + LOOSE;

fn stack_bits(s: &State) -> ([u8; CONT_CAP], [u64; CONT_CAP]) {
    let mut kinds = [0xffu8; CONT_CAP];
    let mut owed = [0u64; CONT_CAP];
    for (i, c) in s.stack().into_iter().enumerate() {
        if let Some(c) = c {
            kinds[i] = c.tag();
            owed[i] = c.owed_hexes();
        }
    }
    (kinds, owed)
}

fn stack_from_row(row: &[u8]) -> ([u8; CONT_CAP], [u64; CONT_CAP]) {
    let mut kinds = [0u8; CONT_CAP];
    let mut owed = [0u64; CONT_CAP];
    kinds.copy_from_slice(&row[ROW_STACK_KIND..ROW_STACK_KIND + CONT_CAP]);
    for d in 0..CONT_CAP {
        let o = ROW_STACK_OWED + d * 8;
        owed[d] = u64::from_le_bytes(row[o..o + 8].try_into().unwrap());
    }
    (kinds, owed)
}

pub fn pack_row(s: &State, ctx: &Ctx, out: &mut [u8]) {
    debug_assert_eq!(out.len(), ROW_BYTES);
    for t in 0..2 * NSLOT {
        out[ROW_IDS + t] = ctx.slots[t / NSLOT][t % NSLOT];
    }
    for h in 0..N_HEXES {
        out[ROW_HEX_OWNER + h] = s.hex_owner[h];
        out[ROW_HEX_SLOT + h] = if s.hex_owner[h] == NONE {
            NONE
        } else {
            ctx.slot_of[s.hex_owner[h] as usize][s.hex_type[h] as usize] as u8
        };
        out[ROW_HEX_HEIGHT + h] = s.hex_height[h];
        out[ROW_HEX_MARKER + h] = s.loc_marker[h];
    }
    let actor = s.to_act() as usize;
    for p in 0..2usize {
        let res = reserve(s, p as u8, ctx);
        let seen = u8::from(p == actor);
        for k in 0..NSLOT {
            let u = ctx.slots[p][k] as usize;
            let z = &s.zones[p];
            out[ROW_PILES + (p * NSLOT + k) * PILE_COUNTS..][..PILE_COUNTS].copy_from_slice(&[
                res[k],
                seen * z[Z_HAND][u],
                seen * z[Z_FACEDOWN][u],
                seen * z[Z_INFLIGHT][u],
                z[Z_FACEUP][u],
                z[Z_SUPPLY][u],
                z[Z_ELIM][u],
            ]);
        }
        out[ROW_HAND_SIZE + p] = s.hand_size(p as u8);
        out[ROW_FD_SIZE + p] = s.zones[p][Z_FACEDOWN].iter().sum();
        out[ROW_BAG_SIZE + p] = s.bag_size(p as u8);
    }
    out[ROW_INITIATIVE] = s.initiative;
    out[ROW_INIT_MOVED] = s.initiative_moved as u8;
    out[ROW_TO_ACT] = s.to_act();
    out[ROW_WP] = s.wp_v2_triggered as u8;
    let (kinds, owed) = stack_bits(s);
    out[ROW_STACK_KIND..ROW_STACK_KIND + CONT_CAP].copy_from_slice(&kinds);
    for d in 0..CONT_CAP {
        let o = ROW_STACK_OWED + d * 8;
        out[o..o + 8].copy_from_slice(&owed[d].to_le_bytes());
    }
}

pub fn mover_row(src: &[u8], dst: &mut [u8]) {
    if src[ROW_TO_ACT] == 1 {
        mirror_row(src, dst);
    } else {
        dst.copy_from_slice(src);
    }
}

pub fn mirror_row(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), ROW_BYTES);
    debug_assert_eq!(dst.len(), ROW_BYTES);
    let flip = |v: u8| match v {
        0 => 1,
        1 => 0,
        other => other,
    };
    dst.copy_from_slice(src);
    for h in 0..N_HEXES {
        let m = crate::state::mirror_hex(h);
        dst[ROW_HEX_OWNER + h] = flip(src[ROW_HEX_OWNER + m]);
        dst[ROW_HEX_SLOT + h] = src[ROW_HEX_SLOT + m];
        dst[ROW_HEX_HEIGHT + h] = src[ROW_HEX_HEIGHT + m];
        dst[ROW_HEX_MARKER + h] = flip(src[ROW_HEX_MARKER + m]);
    }
    for k in 0..NSLOT {
        dst[ROW_IDS + k] = src[ROW_IDS + NSLOT + k];
        dst[ROW_IDS + NSLOT + k] = src[ROW_IDS + k];
    }
    let piles = NSLOT * PILE_COUNTS;
    for k in 0..piles {
        dst[ROW_PILES + k] = src[ROW_PILES + piles + k];
        dst[ROW_PILES + piles + k] = src[ROW_PILES + k];
    }
    for at in [ROW_HAND_SIZE, ROW_FD_SIZE, ROW_BAG_SIZE] {
        dst[at] = src[at + 1];
        dst[at + 1] = src[at];
    }
    dst[ROW_INITIATIVE] = flip(src[ROW_INITIATIVE]);
    dst[ROW_TO_ACT] = flip(src[ROW_TO_ACT]);
    for d in 0..CONT_CAP {
        let at = ROW_STACK_OWED + d * 8;
        dst[at..at + 8].fill(0);
        for h in 0..N_HEXES {
            if src[at + h / 8] >> (h % 8) & 1 == 1 {
                let m = crate::state::mirror_hex(h);
                dst[at + m / 8] |= 1 << (m % 8);
            }
        }
    }
}

pub fn expand_row(row: &[u8], out: &mut [f32]) {
    debug_assert_eq!(row.len(), ROW_BYTES);
    debug_assert_eq!(out.len(), PUBFEAT);
    let hex = |at: usize| -> &[u8; N_HEXES] { row[at..at + N_HEXES].try_into().unwrap() };
    let (hex_owner, hex_slot) = (hex(ROW_HEX_OWNER), hex(ROW_HEX_SLOT));
    let (hex_height, hex_marker) = (hex(ROW_HEX_HEIGHT), hex(ROW_HEX_MARKER));
    let piles = &row[ROW_PILES..ROW_PILES + 2 * NSLOT * PILE_COUNTS];
    let ids = &row[ROW_IDS..ROW_IDS + 2 * NSLOT];
    let mut markers_hand = [6u8; 2];
    for h in 0..N_HEXES {
        if hex_marker[h] != NONE {
            markers_hand[hex_marker[h] as usize] -= 1;
        }
    }
    let initiative = if row[ROW_INITIATIVE] > 1 { 0 } else { row[ROW_INITIATIVE] };
    let (kinds, owed) = stack_from_row(row);
    out.fill(0.0);
    let bd = board();

    let mut i = 0;
    for h in 0..N_HEXES {
        let owner = hex_owner[h];
        if owner != NONE {
            out[i + owner as usize] = 1.0;
            out[i + 2] = hex_height[h] as f32 / 5.0;
            if hex_slot[h] != NONE {
                out[i + HEX_FACTS + owner as usize * NSLOT + hex_slot[h] as usize] = 1.0;
            }
        }
        if hex_marker[h] != NONE {
            out[i + 3 + hex_marker[h] as usize] = 1.0;
        }
        out[i + 5] = bd.is_location[h] as u8 as f32;
        for d in 0..CONT_CAP {
            if owed[d] & (1u64 << h) != 0 {
                out[i + 6 + d] = 1.0;
            }
        }
        i += HEX_CH;
    }

    for &x in piles {
        out[i] = x as f32 / 5.0;
        i += 1;
    }

    for t in 0..2 * NSLOT {
        write_card_features(ids[t], &mut out[i..i + CARD_FEATS]);
        i += CARD_FEATS;
    }

    for p in 0..2usize {
        out[i] = markers_hand[p] as f32 / 6.0;
        out[i + 1] = (6 - markers_hand[p]) as f32 / 6.0;
        out[i + 2] = row[ROW_HAND_SIZE + p] as f32 / 3.0;
        out[i + 3] = row[ROW_FD_SIZE + p] as f32 / MAX_COINS;
        out[i + 4] = row[ROW_BAG_SIZE + p] as f32 / MAX_COINS;
        out[i + 5] = (initiative == p as u8) as u8 as f32;
        i += PLAYER_SCALARS;
    }

    out[i] = row[ROW_WP] as f32;
    out[i + 1] = row[ROW_INIT_MOVED] as f32;
    out[i + 2] = (row[ROW_TO_ACT] == 0) as u8 as f32;
    for d in 0..CONT_CAP {
        let k = kinds[d];
        if (k as usize) < PENDING_KINDS {
            out[i + 3 + d * PENDING_KINDS + k as usize] = 1.0;
        }
    }
    debug_assert_eq!(i + GLOBAL_SCALARS, PUBFEAT);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selfplay::resolve_chance;
    use crate::state::WHITE;

    fn toy(worlds: &[(State, Vec<Action>)]) -> Vec<f64> {
        worlds
            .iter()
            .flat_map(|(s, acts)| {
                let r: u32 = s.zones[WHITE as usize].as_flattened().iter().enumerate().map(|(i, &n)| i as u32 * n as u32).sum();
                acts.iter().map(move |a| (1 + (r + a.encode()) % 7) as f64)
            })
            .collect()
    }

    #[test]
    fn the_filter_matches_exact_enumeration_through_a_partial_reshuffle() {
        for likelihood in [toy, ways] {
            let mut start = State::from_draft(&[17, 12, 4, 9], &[1, 3, 8, 16], WHITE);
            let z = &mut start.zones[WHITE as usize];
            for u in 0..N_UNITS {
                let n = z[Z_BAG][u] / 2;
                z[Z_BAG][u] -= n;
                z[Z_FACEUP][u] += n;
            }
            let (mut rng, mut belief, mut s, mut seen) = (Rng::new(3), Belief::new(&start, WHITE), start, start);
            let (mut worlds, mut reshuffled, mut after) = (vec![(start, 1.0)], false, 0);
            while after < 3 {
                let (p, chance) = (s.to_act(), s.is_chance());
                let a = if chance {
                    reshuffled |= p == WHITE && s.bag_size(p) == 0 && s.zones[p as usize][Z_FACEDOWN].iter().sum::<u8>() > 0;
                    resolve_chance(&mut s, p, &mut rng)
                } else {
                    let acts = s.legal_actions();
                    let hidden: Vec<Action> = acts.iter().copied().filter(|a| obs_key(a) != a.encode()).collect();
                    let pool = if p == WHITE && !hidden.is_empty() { hidden } else { acts };
                    let a = pool[rng.below(pool.len())];
                    s.apply_inplace(a);
                    a
                };
                if p == WHITE {
                    let saw = |b: &Action| chance || obs_key(b) == obs_key(&a);
                    seen = belief.update(&seen, saw, likelihood).unwrap();
                    worlds = worlds
                        .into_iter()
                        .flat_map(|(w, x)| {
                            let acts = w.legal_actions();
                            let odds = likelihood(&[(w, acts.clone())]);
                            acts.into_iter().zip(odds).filter(|(b, _)| saw(b)).map(move |(b, l)| (w.apply(b), x * l))
                        })
                        .collect();
                } else {
                    seen.apply_inplace(a);
                    worlds.iter_mut().for_each(|(w, _)| w.apply_inplace(a));
                }
                after += reshuffled as usize;
            }
            let mut exact = BTreeMap::new();
            for (w, x) in &worlds {
                *exact.entry(layout(w, WHITE)).or_insert(0.0) += x;
            }
            let total: f64 = exact.values().sum();
            assert!(exact.len() > 1 && exact.contains_key(&layout(&s, WHITE)));
            assert_eq!(belief.layouts, exact.keys().copied().collect::<Vec<_>>());
            for (w, x) in belief.weights.iter().zip(exact.values()) {
                assert!((w - x / total).abs() < 1e-12, "{w} is not {}", x / total);
            }
        }
    }

    #[test]
    fn a_revealed_coin_weighs_each_hand_by_its_copies() {
        let mut s = State::from_draft(&[17, 12, 4, 9], &[1, 3, 8, 16], WHITE);
        let mut belief = Belief::new(&s, WHITE);
        while s.is_chance() {
            s = match s.to_act() {
                WHITE => belief.update(&s, |_| true, ways).unwrap(),
                _ => s.apply(s.legal_actions()[0]),
            };
        }
        let [t, x, y] = [17, 12, 4].map(|id| crate::units::index_of_id(id).unwrap());
        let deploy = Action::Deploy { unit: t, hex: board().location_hexes[0] };
        belief.update(&s, |a| *a == deploy, ways).unwrap();
        let weight = |coins: &[u8]| {
            let mut hand = [0; N_UNITS];
            coins.iter().for_each(|&c| hand[c as usize] += 1);
            let holds = |k: &Layout| {
                let mut w = s;
                put(&mut w, WHITE, k);
                w.zones[WHITE as usize][Z_HAND] == hand
            };
            belief.layouts.iter().zip(&belief.weights).find(|(k, _)| holds(k)).unwrap().1
        };
        assert!((weight(&[t, x]) / weight(&[x, y]) - 0.5).abs() < 1e-12);
    }
}
