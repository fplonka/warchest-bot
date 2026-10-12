use crate::actions::{Action, N_KINDS};
use crate::board::{bits, board, mask, NONE, N_HEXES};
use crate::rng::{cumulative, Rng};
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

type Layout = [[u8; NSLOT]; HIDDEN.len()];

pub struct Belief {
    p: u8,
    slots: [u8; NSLOT],
    layouts: Vec<Layout>,
    weights: Vec<f64>,
    cdf: Vec<f64>,
}

impl Belief {
    pub fn new(s: &State, p: u8) -> Belief {
        let slots = Ctx::new(s).slots[p as usize];
        let mut belief = Belief { p, slots, layouts: Vec::new(), weights: vec![1.0], cdf: vec![1.0] };
        belief.layouts.push(belief.layout(s));
        belief
    }

    fn layout(&self, s: &State) -> Layout {
        let z = &s.zones[self.p as usize];
        HIDDEN.map(|h| self.slots.map(|u| z[h][u as usize]))
    }

    fn put(&self, s: &mut State, k: &Layout) {
        let z = &mut s.zones[self.p as usize];
        for (&h, counts) in HIDDEN.iter().zip(k) {
            for (&u, &n) in self.slots.iter().zip(counts) {
                z[h][u as usize] = n;
            }
        }
    }

    pub fn deal(&self, s: &mut State, rng: &mut Rng) {
        self.put(s, &self.layouts[rng.pick(&self.cdf)]);
    }

    pub fn update(
        &mut self,
        s: &State,
        seen: impl Fn(&Action) -> bool,
        likelihood: impl FnOnce(&[(State, Vec<Action>)]) -> Vec<f64>,
    ) -> Option<State> {
        let mut acts = Vec::new();
        let (worlds, weights): (Vec<_>, Vec<f64>) = self
            .layouts
            .iter()
            .zip(&self.weights)
            .filter_map(|(k, &x)| {
                let mut w = *s;
                self.put(&mut w, k);
                w.legal_actions_into(&mut acts);
                acts.iter().any(&seen).then(|| ((w, acts.clone()), x))
            })
            .unzip();
        let mut odds = likelihood(&worlds).into_iter();
        let (mut next, mut last) = (Vec::new(), None);
        for ((w, acts), x) in worlds.iter().zip(weights) {
            for (a, l) in acts.iter().zip(odds.by_ref()) {
                if seen(a) {
                    next.push((self.layout(&w.apply(*a)), x * l));
                    last = Some((w, a));
                }
            }
        }
        let (w, a) = last?;
        next.sort_by_key(|e| e.0);
        (self.layouts, self.weights) =
            next.chunk_by(|a, b| a.0 == b.0).map(|run| (run[0].0, run.iter().map(|e| e.1).sum::<f64>())).unzip();
        let total: f64 = self.weights.iter().sum();
        self.weights.iter_mut().for_each(|x| *x /= total);
        self.cdf = cumulative(self.weights.iter().copied());
        Some(w.apply(*a))
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
    let out: &mut [u8; ROW_BYTES] = out.try_into().expect("a row is ROW_BYTES long");
    out[ROW_IDS..ROW_HEX_OWNER].copy_from_slice(ctx.slots.as_flattened());
    out[ROW_HEX_OWNER..ROW_HEX_SLOT].copy_from_slice(&s.hex_owner);
    out[ROW_HEX_SLOT..ROW_HEX_HEIGHT].fill(NONE);
    for h in bits(mask(N_HEXES, |h| s.hex_owner[h] != NONE)) {
        out[ROW_HEX_SLOT + h] = ctx.slot_of[s.hex_owner[h] as usize][s.hex_type[h] as usize] as u8;
    }
    out[ROW_HEX_HEIGHT..ROW_HEX_MARKER].copy_from_slice(&s.hex_height);
    out[ROW_HEX_MARKER..ROW_PILES].copy_from_slice(&s.loc_marker);
    let actor = s.to_act();
    for p in 0..2 {
        let (z, seen) = (&s.zones[p], u8::from(p == actor as usize));
        for (k, &u) in ctx.slots[p].iter().enumerate() {
            let u = u as usize;
            out[ROW_PILES + (p * NSLOT + k) * PILE_COUNTS..][..PILE_COUNTS].copy_from_slice(&[
                HIDDEN.iter().map(|&h| z[h][u]).sum(),
                seen * z[Z_HAND][u],
                seen * z[Z_FACEDOWN][u],
                seen * z[Z_INFLIGHT][u],
                z[Z_FACEUP][u],
                z[Z_SUPPLY][u],
                z[Z_ELIM][u],
            ]);
        }
        out[ROW_HAND_SIZE + p] = z[Z_HAND].iter().sum();
        out[ROW_FD_SIZE + p] = z[Z_FACEDOWN].iter().sum();
        out[ROW_BAG_SIZE + p] = z[Z_BAG].iter().sum();
    }
    out[ROW_INITIATIVE] = s.initiative;
    out[ROW_INIT_MOVED] = s.initiative_moved as u8;
    out[ROW_TO_ACT] = actor;
    out[ROW_WP] = s.wp_v2_triggered as u8;
    out[ROW_STACK_KIND..ROW_STACK_OWED].fill(0xff);
    out[ROW_STACK_OWED..].fill(0);
    for (d, c) in std::iter::once(&s.pending).chain(s.conts.iter()).take(CONT_CAP).enumerate() {
        out[ROW_STACK_KIND + d] = c.tag();
        out[ROW_STACK_OWED + d * 8..][..8].copy_from_slice(&c.owed_hexes().to_le_bytes());
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
    let src: &[u8; ROW_BYTES] = src.try_into().expect("a row is ROW_BYTES long");
    let dst: &mut [u8; ROW_BYTES] = dst.try_into().expect("a row is ROW_BYTES long");
    let flip = |v: u8| match v {
        0 => 1,
        1 => 0,
        other => other,
    };
    let mirror = &board().mirror;
    dst.copy_from_slice(src);
    for (h, &m) in mirror.iter().enumerate() {
        let m = m as usize;
        dst[ROW_HEX_OWNER + h] = flip(src[ROW_HEX_OWNER + m]);
        dst[ROW_HEX_SLOT + h] = src[ROW_HEX_SLOT + m];
        dst[ROW_HEX_HEIGHT + h] = src[ROW_HEX_HEIGHT + m];
        dst[ROW_HEX_MARKER + h] = flip(src[ROW_HEX_MARKER + m]);
    }
    dst[ROW_IDS..ROW_HEX_OWNER].rotate_left(NSLOT);
    dst[ROW_PILES..ROW_HAND_SIZE].rotate_left(NSLOT * PILE_COUNTS);
    for at in [ROW_HAND_SIZE, ROW_FD_SIZE, ROW_BAG_SIZE] {
        dst.swap(at, at + 1);
    }
    dst[ROW_INITIATIVE] = flip(src[ROW_INITIATIVE]);
    dst[ROW_TO_ACT] = flip(src[ROW_TO_ACT]);
    for owed in dst[ROW_STACK_OWED..].chunks_exact_mut(8) {
        let mask = u64::from_le_bytes((&*owed).try_into().expect("an owed mask is 8 bytes"));
        owed.copy_from_slice(&bits(mask).fold(0u64, |m, h| m | 1 << mirror[h]).to_le_bytes());
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
    use std::collections::BTreeMap;

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
                *exact.entry(belief.layout(w)).or_insert(0.0) += x;
            }
            let total: f64 = exact.values().sum();
            assert!(exact.len() > 1 && exact.contains_key(&belief.layout(&s)));
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
                belief.put(&mut w, k);
                w.zones[WHITE as usize][Z_HAND] == hand
            };
            belief.layouts.iter().zip(&belief.weights).find(|(k, _)| holds(k)).unwrap().1
        };
        assert!((weight(&[t, x]) / weight(&[x, y]) - 0.5).abs() < 1e-12);
    }
}
