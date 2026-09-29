use std::collections::BTreeMap;

use crate::actions::{Action, N_KINDS};
use crate::board::{board, NONE, N_HEXES};
use crate::rng::Rng;
use crate::state::{
    State, CONT_CAP, N_ZONES, PENDING_KINDS, Z_BAG, Z_ELIM, Z_FACEDOWN, Z_FACEUP, Z_HAND, Z_INFLIGHT, Z_SUPPLY,
};
use crate::units::{write_card_features, CARD_FEATS, N_UNITS};

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
    let n = hidden(s, p);
    ctx.slots[p as usize].map(|u| n[u as usize])
}

fn hidden(s: &State, p: u8) -> [u8; N_UNITS] {
    std::array::from_fn(|u| HIDDEN.iter().map(|&z| s.zones[p as usize][z][u]).sum())
}

fn spread(n: [u8; N_UNITS]) -> Vec<u8> {
    (0..N_UNITS as u8)
        .flat_map(|u| std::iter::repeat_n(u, n[u as usize] as usize))
        .collect()
}

pub fn unseen(s: &State, p: u8) -> Vec<u8> {
    spread(hidden(s, p))
}

pub fn deal(s: &mut State, p: u8, coins: &[u8]) {
    let z = &mut s.zones[p as usize];
    let mut rest = coins;
    for zone in HIDDEN {
        let (dealt, left) = rest.split_at(z[zone].iter().sum::<u8>() as usize);
        z[zone] = [0; N_UNITS];
        for &u in dealt {
            z[zone][u as usize] += 1;
        }
        rest = left;
    }
    debug_assert!(rest.is_empty(), "every unseen coin lands in a hidden zone");
}

const ANCHOR: usize = N_UNITS;
type Layout = [[u8; N_UNITS + 1]; N_ZONES];

fn free(k: &Layout, hidden: [u8; N_UNITS]) -> [u8; N_UNITS] {
    std::array::from_fn(|u| hidden[u] - HIDDEN.iter().map(|&z| k[z][u]).sum::<u8>())
}

pub struct Belief {
    layouts: Vec<Layout>,
    weights: Vec<f64>,
}

impl Belief {
    pub fn new(s: &State, p: u8) -> Belief {
        let mut k = [[0; N_UNITS + 1]; N_ZONES];
        for z in HIDDEN {
            k[z][ANCHOR] = s.zones[p as usize][z].iter().sum();
        }
        Belief { layouts: vec![k], weights: vec![1.0] }
    }

    pub fn observe(&mut self, before: &State, after: &State, p: u8) {
        let size = |s: &State, z: usize| s.zones[p as usize][z].iter().sum::<u8>();
        let held = [Z_HAND, Z_INFLIGHT];
        if let Some(to) = held.into_iter().find(|&z| size(after, z) > size(before, z)) {
            if size(before, Z_BAG) == 0 {
                let faceup = before.zones[p as usize][Z_FACEUP];
                for k in &mut self.layouts {
                    let down = std::mem::take(&mut k[Z_FACEDOWN]);
                    k[Z_BAG] = std::array::from_fn(|c| down[c] + faceup.get(c).unwrap_or(&0));
                }
            }
            self.take(Z_BAG, Some(to), |_, _| 1.0);
        } else if let Some(from) = held.into_iter().find(|&z| size(after, z) < size(before, z)) {
            let (was, now) = (hidden(before, p), hidden(after, p));
            match (0..N_UNITS).find(|&u| now[u] < was[u]) {
                None => self.take(from, Some(Z_FACEDOWN), |_, _| 1.0),
                Some(t) => self.take(from, None, |k, c| match c {
                    _ if c == t => 1.0,
                    ANCHOR => {
                        let f = free(k, was);
                        f[t] as f64 / f.iter().sum::<u8>() as f64
                    }
                    _ => 0.0,
                }),
            }
        }
        if HIDDEN.iter().filter(|&&z| size(after, z) > 0).count() <= 1 {
            *self = Belief::new(after, p);
        }
    }

    fn take(&mut self, from: usize, to: Option<usize>, fit: impl Fn(&Layout, usize) -> f64) {
        let mut next = BTreeMap::new();
        for (k, w) in self.layouts.iter().zip(&self.weights) {
            for c in 0..=N_UNITS {
                let w = w * k[from][c] as f64 * fit(k, c);
                if w > 0.0 {
                    let mut k = *k;
                    k[from][c] -= 1;
                    if let Some(to) = to {
                        k[to][c] += 1;
                    }
                    *next.entry(k).or_insert(0.0) += w;
                }
            }
        }
        let total: f64 = next.values().sum();
        (self.layouts, self.weights) = next.into_iter().map(|(k, w)| (k, w / total)).unzip();
    }

    pub fn deal(&self, s: &mut State, p: u8, rng: &mut Rng) {
        let k = &self.layouts[rng.weighted_index(&self.weights)];
        let mut coins = spread(free(k, hidden(s, p)));
        rng.shuffle(&mut coins);
        let mut coins = coins.into_iter();
        for z in HIDDEN {
            let zone = &mut s.zones[p as usize][z];
            zone.copy_from_slice(&k[z][..N_UNITS]);
            for u in coins.by_ref().take(k[z][ANCHOR] as usize) {
                zone[u as usize] += 1;
            }
        }
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
    use crate::rng::Rng;
    use crate::state::{N_ZONES, WHITE};
    use crate::units::{ARCHER, PIKEMAN, SWORDSMAN};
    use std::collections::HashMap;

    #[derive(Clone, Copy)]
    enum Ev {
        Draw,
        Show(u8),
        Hide,
    }

    fn branches(s: &State, e: Ev) -> Vec<(State, f64)> {
        let mut s = *s;
        let z = &mut s.zones[0];
        if matches!(e, Ev::Draw) && z[Z_BAG].iter().all(|&n| n == 0) {
            for u in 0..N_UNITS {
                z[Z_BAG][u] = std::mem::take(&mut z[Z_FACEUP][u]) + std::mem::take(&mut z[Z_FACEDOWN][u]);
            }
        }
        let (from, to) = match e {
            Ev::Draw => (Z_BAG, Z_HAND),
            Ev::Show(_) => (Z_HAND, Z_FACEUP),
            Ev::Hide => (Z_HAND, Z_FACEDOWN),
        };
        (0..N_UNITS)
            .filter(|&u| s.zones[0][from][u] > 0 && !matches!(e, Ev::Show(t) if t as usize != u))
            .map(|u| {
                let mut t = s;
                t.zones[0][from][u] -= 1;
                t.zones[0][to][u] += 1;
                (t, s.zones[0][from][u] as f64)
            })
            .collect()
    }

    fn posterior(s: &State, events: &[Ev]) -> (State, Belief, HashMap<[[u8; N_UNITS]; N_ZONES], f64>) {
        let mut worlds = vec![(*s, 1.0)];
        let (mut truth, mut belief) = (*s, Belief::new(s, WHITE));
        for &e in events {
            let next = branches(&truth, e)[0].0;
            belief.observe(&truth, &next, WHITE);
            truth = next;
            worlds = worlds
                .iter()
                .flat_map(|(w, p)| branches(w, e).into_iter().map(move |(v, q)| (v, p * q)))
                .collect();
        }
        let mut exact = HashMap::new();
        let total: f64 = worlds.iter().map(|w| w.1).sum();
        for (w, p) in worlds {
            *exact.entry(w.zones[0]).or_insert(0.0) += p / total;
        }
        (truth, belief, exact)
    }

    fn matches_brute_force(s: &State, events: &[Ev]) {
        let (truth, belief, exact) = posterior(s, events);
        let (mut rng, n) = (Rng::new(11), 100_000);
        let mut got: HashMap<[[u8; N_UNITS]; N_ZONES], f64> = HashMap::new();
        for _ in 0..n {
            let mut w = truth;
            belief.deal(&mut w, WHITE, &mut rng);
            *got.entry(w.zones[0]).or_insert(0.0) += 1.0 / n as f64;
        }
        for w in exact.keys().chain(got.keys()) {
            let (a, b) = (exact.get(w).copied().unwrap_or(0.0), got.get(w).copied().unwrap_or(0.0));
            assert!((a - b).abs() < 0.01, "world {w:?}: exact {a}, dealt {b}");
        }
    }

    #[test]
    fn a_partial_reshuffle_keeps_the_old_coin_in_hand() {
        let mut s = State::blank(WHITE);
        s.add_zone(WHITE, Z_BAG, SWORDSMAN, 1);
        s.add_zone(WHITE, Z_FACEUP, PIKEMAN, 5);
        let (truth, belief, _) = posterior(&s, &[Ev::Draw, Ev::Draw, Ev::Draw]);
        let mut rng = Rng::new(1);
        for _ in 0..1000 {
            let mut w = truth;
            belief.deal(&mut w, WHITE, &mut rng);
            assert_eq!(w.zones[0][Z_HAND][SWORDSMAN as usize], 1);
            assert_eq!(w.zones[0][Z_HAND][PIKEMAN as usize], 2);
        }
    }

    #[test]
    fn a_reveal_after_a_partial_reshuffle_matches_brute_force() {
        let mut s = State::blank(WHITE);
        s.add_zone(WHITE, Z_BAG, SWORDSMAN, 1);
        s.add_zone(WHITE, Z_BAG, ARCHER, 2);
        s.add_zone(WHITE, Z_BAG, PIKEMAN, 1);
        s.add_zone(WHITE, Z_FACEUP, PIKEMAN, 2);
        s.add_zone(WHITE, Z_FACEUP, SWORDSMAN, 1);
        let (draw, hide) = (Ev::Draw, Ev::Hide);
        matches_brute_force(&s, &[draw, draw, draw, Ev::Show(ARCHER), hide, draw]);
        matches_brute_force(&s, &[draw, draw, draw, hide, hide, draw, draw, draw, Ev::Show(PIKEMAN), hide]);
    }

    #[test]
    fn a_dealt_world_places_each_unseen_coin_by_zone_size() {
        let mut s = State::blank(WHITE);
        s.add_zone(WHITE, Z_BAG, SWORDSMAN, 3);
        s.add_zone(WHITE, Z_FACEDOWN, SWORDSMAN, 2);
        s.add_zone(WHITE, Z_HAND, PIKEMAN, 1);
        let (mut rng, mut coins, n) = (Rng::new(5), unseen(&s, WHITE), 60_000);
        let mut seen = [0usize; N_ZONES];
        for _ in 0..n {
            rng.shuffle(&mut coins);
            let mut w = s;
            deal(&mut w, WHITE, &coins);
            assert_eq!(unseen(&w, WHITE), unseen(&s, WHITE));
            for z in HIDDEN {
                assert_eq!(w.zones[0][z].iter().sum::<u8>(), s.zones[0][z].iter().sum::<u8>());
                seen[z] += w.zones[0][z][PIKEMAN as usize] as usize;
            }
        }
        for (z, want) in [(Z_HAND, 1.0 / 6.0), (Z_FACEDOWN, 2.0 / 6.0), (Z_BAG, 3.0 / 6.0)] {
            let got = seen[z] as f64 / n as f64;
            assert!((got - want).abs() < 0.01, "zone {z}: {got} is not {want}");
        }
    }
}
