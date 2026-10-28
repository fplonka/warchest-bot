use std::path::Path;

use warchest::actions::N_KINDS;
use warchest::board::{bits, board, mask, NONE, N_HEXES};
use warchest::state::{CONT_CAP, PENDING_KINDS};
use warchest::units::N_UNITS;

use crate::pbs::{
    NSLOT, PILE_COUNTS, ROW_BAG_SIZE, ROW_BYTES, ROW_FD_SIZE, ROW_HAND_SIZE, ROW_HEX_HEIGHT, ROW_HEX_MARKER, ROW_HEX_OWNER,
    ROW_HEX_SLOT, ROW_IDS, ROW_INITIATIVE, ROW_INIT_MOVED, ROW_PILES, ROW_STACK_KIND, ROW_STACK_OWED, ROW_TO_ACT, ROW_WP,
};
use crate::search::{Answer, Oracle, Query, ACT_BYTES};

pub const H: usize = 256;
pub const CLIP: f32 = 1.0;
pub const MAX_ACTIVE: usize = 128;
pub const MAX_MOVE: usize = 8;
const Q: i16 = 127;
const QQ: f32 = (Q * Q) as f32;
const LANES: usize = 128;
const HEIGHT: usize = 5;
const ZONES: usize = 7;
const COINS: usize = 22;

const STACK: usize = 0;
const PILE: usize = STACK + 2 * N_UNITS * N_HEXES * HEIGHT;
const MARKER: usize = PILE + 2 * ZONES * N_UNITS * HEIGHT;
const COUNT: usize = MARKER + 2 * N_HEXES;
const FLAG: usize = COUNT + 2 * 4 * COINS;
const CONT: usize = FLAG + 4;
const OWED: usize = CONT + CONT_CAP * PENDING_KINDS;
pub const FEATURES: usize = OWED + CONT_CAP * N_HEXES;

const KIND_COIN: usize = 0;
const KIND_HEX: usize = KIND_COIN + N_KINDS * (N_UNITS + 1);
const COIN_HEX: usize = KIND_HEX + N_KINDS * 3 * N_HEXES;
const RECRUIT: usize = COIN_HEX + (N_UNITS + 1) * 3 * N_HEXES;
pub const MOVES: usize = RECRUIT + N_UNITS;

pub struct Active<const N: usize> {
    at: [u16; N],
    len: usize,
}

impl<const N: usize> Active<N> {
    fn new() -> Self {
        Active { at: [0; N], len: 0 }
    }

    fn push(&mut self, f: usize, on: bool) {
        self.at[self.len] = f as u16;
        self.len += on as usize;
    }
}

impl<const N: usize> std::ops::Deref for Active<N> {
    type Target = [u16];
    fn deref(&self) -> &[u16] {
        &self.at[..self.len]
    }
}

type Row = [u8; ROW_BYTES];

fn orient(row: &Row) -> impl Fn(usize) -> usize {
    let (flip, mirror) = (row[ROW_TO_ACT] == 1, &board().mirror);
    move |h| if flip { mirror[h] as usize } else { h }
}

fn present(bytes: &[u8]) -> impl Iterator<Item = usize> + '_ {
    bits(mask(bytes.len(), |i| bytes[i] != NONE))
}

pub fn features(row: &Row) -> Active<MAX_ACTIVE> {
    let mut out = Active::new();
    let me = row[ROW_TO_ACT];
    let side = |p: u8| (p != me) as usize;
    let hex = orient(row);
    let unit = |p: usize, k: usize| row[ROW_IDS + p * NSLOT + k] as usize;
    for h in present(&row[ROW_HEX_OWNER..][..N_HEXES]) {
        let owner = row[ROW_HEX_OWNER + h];
        let u = unit(owner as usize, row[ROW_HEX_SLOT + h] as usize);
        out.push(STACK + ((side(owner) * N_UNITS + u) * N_HEXES + hex(h)) * HEIGHT + row[ROW_HEX_HEIGHT + h] as usize - 1, true);
    }
    let mut markers = [0u8; 2];
    for h in present(&row[ROW_HEX_MARKER..][..N_HEXES]) {
        let s = side(row[ROW_HEX_MARKER + h]);
        markers[s] += 1;
        out.push(MARKER + s * N_HEXES + hex(h), true);
    }
    for p in 0..2 {
        let s = side(p as u8);
        for k in 0..NSLOT {
            let n = &row[ROW_PILES + (p * NSLOT + k) * PILE_COUNTS..][..PILE_COUNTS];
            let zones = [n[0] - n[1] - n[2] - n[3], n[1], n[2], n[3], n[4], n[5], n[6]];
            let at = PILE + (s * ZONES * N_UNITS + unit(p, k)) * HEIGHT;
            for (z, c) in zones.into_iter().enumerate() {
                out.push(at + z * N_UNITS * HEIGHT + c as usize - 1, c > 0);
            }
        }
        let counts = [markers[s], row[ROW_HAND_SIZE + p], row[ROW_BAG_SIZE + p], row[ROW_FD_SIZE + p]];
        for (what, n) in counts.into_iter().enumerate() {
            out.push(COUNT + (s * 4 + what) * COINS + n as usize, true);
        }
    }
    out.push(FLAG + side(row[ROW_INITIATIVE]), row[ROW_INITIATIVE] < 2);
    out.push(FLAG + 2, row[ROW_INIT_MOVED] == 1);
    out.push(FLAG + 3, row[ROW_WP] == 1);
    let kinds = row[ROW_STACK_KIND..][..CONT_CAP].iter().take_while(|&&k| (k as usize) < PENDING_KINDS);
    for (d, &kind) in kinds.enumerate() {
        out.push(CONT + d * PENDING_KINDS + kind as usize, true);
        for h in bits(u64::from_le_bytes(row[ROW_STACK_OWED + 8 * d..][..8].try_into().unwrap())) {
            out.push(OWED + d * N_HEXES + hex(h), true);
        }
    }
    out
}

pub fn move_features(row: &Row, desc: &[u8]) -> Active<MAX_MOVE> {
    let mut out = Active::new();
    let hex = orient(row);
    let unit = |t: u8| if t == NONE { N_UNITS } else { row[ROW_IDS + t as usize] as usize };
    let (kind, coin) = (desc[0] as usize, unit(desc[1]));
    out.push(KIND_COIN + kind * (N_UNITS + 1) + coin, true);
    for (slot, &h) in desc[3..].iter().enumerate().filter(|&(_, &h)| h != NONE) {
        out.push(KIND_HEX + (kind * 3 + slot) * N_HEXES + hex(h as usize), true);
        out.push(COIN_HEX + (coin * 3 + slot) * N_HEXES + hex(h as usize), true);
    }
    out.push(RECRUIT + unit(desc[2]), desc[2] != NONE);
    out
}

pub const SIZES: [usize; 6] = [FEATURES * H, H, H, 1, MOVES * H, MOVES];

pub struct Nnue {
    ft: Vec<[i8; H]>,
    ft_bias: [i16; H],
    value: [i8; H],
    value_bias: f32,
    policy: Vec<[i8; H]>,
    policy_bias: Vec<f32>,
}

impl Nnue {
    pub fn load(path: &Path) -> Result<Nnue, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let floats = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        Nnue::from_floats(floats).map_err(|e| format!("{}: {e}", path.display()))
    }

    fn from_floats(floats: Vec<f32>) -> Result<Nnue, String> {
        if floats.len() != SIZES.iter().sum::<usize>() {
            return Err(format!("{} weights do not match the nnue's {}", floats.len(), SIZES.iter().sum::<usize>()));
        }
        if let Some(x) = floats.iter().find(|x| !(-CLIP..=CLIP).contains(*x)) {
            return Err(format!("weight {x} is outside the clip {CLIP}"));
        }
        let mut rest = &floats[..];
        let [ft, ft_bias, value, value_bias, policy, policy_bias] = SIZES.map(|n| {
            let (a, b) = rest.split_at(n);
            rest = b;
            a
        });
        let rows = |a: &[f32]| -> Vec<[i8; H]> {
            a.chunks_exact(H).map(|r| std::array::from_fn(|i| (r[i] * Q as f32).round() as i8)).collect()
        };
        Ok(Nnue {
            ft: rows(ft),
            ft_bias: rows(ft_bias)[0].map(i16::from),
            value: rows(value)[0],
            value_bias: value_bias[0],
            policy: rows(policy),
            policy_bias: policy_bias.to_vec(),
        })
    }
}

fn accumulate(table: &[[i8; H]], feats: &[u16], mut acc: [i16; H]) -> [i16; H] {
    for (b, lanes) in acc.chunks_exact_mut(LANES).enumerate() {
        let mut r: [i16; LANES] = (&*lanes).try_into().unwrap();
        for &f in feats {
            for (x, &w) in r.iter_mut().zip(&table[f as usize][b * LANES..][..LANES]) {
                *x += w as i16;
            }
        }
        lanes.copy_from_slice(&r);
    }
    acc
}

fn dot(a: &[i8; H], b: &[i8; H]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

fn softmax(x: &mut [f32]) {
    let top = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    x.iter_mut().for_each(|v| *v = (*v - top).exp());
    let total: f32 = x.iter().sum();
    x.iter_mut().for_each(|v| *v /= total);
}

impl Oracle for Nnue {
    fn eval(&self, q: &Query) -> Answer {
        let mut answer = Answer { value: Vec::with_capacity(q.len()), prior: Vec::with_capacity(q.desc.len() / ACT_BYTES) };
        for i in 0..q.len() {
            let (row, desc) = q.row(i);
            let row = row.try_into().unwrap();
            let hl = accumulate(&self.ft, &features(row), self.ft_bias).map(|x| x.clamp(0, Q) as i8);
            answer.value.push(((dot(&hl, &self.value) as f32 / QQ + self.value_bias) / 2.0).tanh());
            let start = answer.prior.len();
            for d in desc.chunks_exact(ACT_BYTES) {
                let moves = move_features(row, d);
                let logit: i32 = moves.iter().map(|&f| dot(&hl, &self.policy[f as usize])).sum();
                let bias: f32 = moves.iter().map(|&f| self.policy_bias[f as usize]).sum();
                answer.prior.push(logit as f32 / QQ + bias);
            }
            softmax(&mut answer.prior[start..]);
        }
        answer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbs::{mirror_row, Ctx};
    use crate::search::{mirror_desc, uniform};
    use warchest::arena::game;
    use warchest::rng::Rng;

    fn positions() -> Query {
        let (mut rng, mut q) = (Rng::new(1), Query::default());
        for _ in 0..20 {
            let mut s = game(&mut rng);
            let ctx = Ctx::new(&s);
            while !warchest::finished(&s) {
                let acts = s.legal_actions();
                if !s.is_chance() {
                    q.push(&s, &ctx, &acts);
                }
                s.apply_inplace(acts[rng.below(acts.len())]);
            }
        }
        q
    }

    fn sorted(feats: &[u16]) -> Vec<u16> {
        let mut out = feats.to_vec();
        out.sort();
        out
    }

    #[test]
    fn features_stay_in_range_and_ignore_the_mirror() {
        let q = positions();
        let mut most = 0;
        for i in 0..q.len() {
            let (row, desc) = q.row(i);
            let row: &Row = row.try_into().unwrap();
            let mut flipped = [0u8; ROW_BYTES];
            mirror_row(row, &mut flipped);
            let feats = sorted(&features(row));
            assert!(feats.iter().all(|&f| (f as usize) < FEATURES) && feats.windows(2).all(|w| w[0] < w[1]), "{feats:?}");
            assert_eq!(feats, sorted(&features(&flipped)));
            most = most.max(feats.len());
            for d in desc.chunks_exact(ACT_BYTES) {
                let mut m = d.to_vec();
                mirror_desc(&mut m);
                assert!(move_features(row, d).iter().all(|&f| (f as usize) < MOVES));
                assert_eq!(sorted(&move_features(row, d)), sorted(&move_features(&flipped, &m)));
            }
        }
        assert!(most < 96, "{most} features");
    }

    #[test]
    fn a_zero_net_is_the_flat_oracle() {
        let q = positions();
        let answer = Nnue::from_floats(vec![0.0; SIZES.iter().sum()]).unwrap().eval(&q);
        assert!(answer.value.iter().all(|&v| v == 0.0));
        assert_eq!(answer.prior, uniform(&q));
    }
}
