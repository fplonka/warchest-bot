use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
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

const MAX_ACTIVE: usize = 512;
const MAX_MOVE: usize = 8;
pub const STACKS: usize = 8;
pub const L2: usize = 32;
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
const PAIR: usize = OWED + CONT_CAP * N_HEXES;
const NEAR: usize = PAIR + (2 * N_UNITS) * (2 * N_UNITS) * 2;
const TARGETS: usize = 2 * N_UNITS + 1;
const LEGAL: usize = NEAR + 2 * N_UNITS * 3;
pub const FEATURES: usize = LEGAL + N_KINDS * (N_UNITS + 1) * TARGETS;

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

    fn set(mut self) -> Self {
        let at = &mut self.at[..self.len];
        at.sort_unstable();
        let mut n = 0;
        for i in 0..at.len() {
            if n == 0 || at[i] != at[n - 1] {
                at[n] = at[i];
                n += 1;
            }
        }
        self.len = n;
        self
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

pub fn features(row: &Row, desc: &[u8]) -> Active<MAX_ACTIVE> {
    let mut out = Active::new();
    let me = row[ROW_TO_ACT];
    let side = |p: u8| (p != me) as usize;
    let hex = orient(row);
    let unit = |p: usize, k: usize| row[ROW_IDS + p * NSLOT + k] as usize;
    let board = board();
    let mut units = [(0usize, 0usize); N_HEXES];
    let occupied = mask(N_HEXES, |h| row[ROW_HEX_OWNER + h] != NONE);
    for h in bits(occupied) {
        let owner = row[ROW_HEX_OWNER + h];
        let u = unit(owner as usize, row[ROW_HEX_SLOT + h] as usize);
        units[h] = (h, side(owner) * N_UNITS + u);
        out.push(STACK + (units[h].1 * N_HEXES + hex(h)) * HEIGHT + row[ROW_HEX_HEIGHT + h] as usize - 1, true);
    }
    for a in bits(occupied) {
        for b in bits(occupied & !(1 << a)) {
            let d = board.dist[a][b] as usize;
            out.push(PAIR + (units[a].1 * 2 * N_UNITS + units[b].1) * 2 + d - 1, d <= 2);
        }
        for &l in board.location_hexes.iter().filter(|&&l| board.dist[a][l as usize] == 1) {
            let m = row[ROW_HEX_MARKER + l as usize];
            out.push(NEAR + units[a].1 * 3 + if m == NONE { 2 } else { side(m) }, true);
        }
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
    for d in desc.chunks_exact(ACT_BYTES) {
        let coin = if d[1] == NONE { N_UNITS } else { row[ROW_IDS + d[1] as usize] as usize };
        let target = d[3..].iter().rev().find(|&&h| h != NONE).map_or(TARGETS - 1, |&h| {
            let owner = row[ROW_HEX_OWNER + h as usize];
            if owner == NONE { TARGETS - 1 } else { units[h as usize].1 }
        });
        out.push(LEGAL + (d[0] as usize * (N_UNITS + 1) + coin) * TARGETS + target, true);
    }
    out.set()
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

pub fn phase(row: &Row) -> u8 {
    let markers = row[ROW_HEX_MARKER..][..N_HEXES].iter().filter(|&&m| m != NONE).count() as u8;
    let lost: u8 = (0..2 * NSLOT).map(|k| row[ROW_PILES + k * PILE_COUNTS + PILE_COUNTS - 1]).sum();
    markers + lost
}

pub fn sizes(l1: usize, r: usize) -> [usize; 14] {
    let half = l1 / 2;
    [FEATURES * l1, l1, FEATURES * STACKS, STACKS * L2 * half, STACKS * L2, STACKS * L2 * 2 * L2, STACKS * L2, STACKS * 4 * L2, STACKS, r * half, r, MOVES * r, r, MOVES]
}

const FT_ONE: f32 = 256.0;
const FT_CLIP: f32 = 127.0;
const ONE: f32 = 128.0;
const OUT: f32 = ONE * ONE;

pub struct Nnue {
    id: u64,
    l1: usize,
    r: usize,
    thresholds: [u8; STACKS - 1],
    ft: Vec<i16>,
    ft_bias: Vec<i32>,
    psqt: Vec<i32>,
    fc0: Vec<i8>,
    fc0_bias: Vec<i32>,
    fc1: Vec<i8>,
    fc1_bias: Vec<i32>,
    out: Vec<i8>,
    out_bias: Vec<i32>,
    proj: Vec<i8>,
    proj_bias: Vec<i32>,
    emb: Vec<i32>,
    head: Vec<i8>,
    move_bias: Vec<i32>,
}

fn quantize<T: TryFrom<i64>>(a: &[f32], scale: f32, clip: f32) -> Result<Vec<T>, String> {
    a.iter()
        .map(|&x| {
            if x.abs() > clip {
                return Err(format!("weight {x} is outside the clip {clip}"));
            }
            T::try_from((x * scale).round() as i64).map_err(|_| format!("weight {x} does not fit at scale {scale}"))
        })
        .collect()
}

impl Nnue {
    pub fn load(path: &Path) -> Result<Nnue, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let floats = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        Nnue::from_floats(floats).map_err(|e| format!("{}: {e}", path.display()))
    }

    fn from_floats(floats: Vec<f32>) -> Result<Nnue, String> {
        let (dims, rest) = floats.split_at(2 + STACKS - 1);
        let (l1, r) = (dims[0] as usize, dims[1] as usize);
        let sizes = sizes(l1, r);
        if l1 % 2 != 0 || rest.len() != sizes.iter().sum::<usize>() {
            return Err(format!("{} weights do not match an nnue of width {l1} and policy rank {r}", rest.len()));
        }
        let mut rest = rest;
        let [ft, ft_bias, psqt, fc0, fc0_bias, fc1, fc1_bias, out, out_bias, proj, proj_bias, emb, head, move_bias] = sizes.map(|n| {
            let (a, b) = rest.split_at(n);
            rest = b;
            a
        });
        let wide = f32::INFINITY;
        Ok(Nnue {
            id: NETS.fetch_add(1, Ordering::Relaxed),
            l1,
            r,
            thresholds: std::array::from_fn(|i| dims[2 + i] as u8),
            ft: quantize(ft, FT_ONE, FT_CLIP)?,
            ft_bias: quantize(ft_bias, FT_ONE, FT_CLIP)?,
            psqt: quantize(psqt, OUT, wide)?,
            fc0: quantize(fc0, ONE, 127.0 / ONE)?,
            fc0_bias: quantize(fc0_bias, OUT, wide)?,
            fc1: quantize(fc1, 64.0, 127.0 / 64.0)?,
            fc1_bias: quantize(fc1_bias, 64.0 * ONE, wide)?,
            out: quantize(out, ONE, 127.0 / ONE)?,
            out_bias: quantize(out_bias, OUT, wide)?,
            proj: quantize(proj, ONE, 127.0 / ONE)?,
            proj_bias: quantize(proj_bias, OUT, wide)?,
            emb: quantize(emb, OUT, wide)?,
            head: quantize(head, ONE, 127.0 / ONE)?,
            move_bias: quantize(move_bias, OUT, wide)?,
        })
    }

    fn stack(&self, row: &Row) -> usize {
        let p = phase(row);
        self.thresholds.iter().filter(|&&t| p >= t).count()
    }

    fn refresh(&self, bits: Bits) -> Accumulator {
        let mut a = Accumulator { net: self.id, used: 0, bits, acc: self.ft_bias.clone(), psqt: [0; STACKS] };
        for (w, &x) in bits.iter().enumerate() {
            set(x).for_each(|b| a.add(self, w * 64 + b, 1));
        }
        a
    }

    fn accumulate(&self, feats: &[u16]) -> (Vec<i8>, [i32; STACKS]) {
        let mut bits = [0u64; WORDS];
        feats.iter().for_each(|&f| bits[f as usize / 64] |= 1 << (f % 64));
        CACHE.with(|cache| {
            let (clock, cache) = &mut *cache.borrow_mut();
            *clock += 1;
            let near = cache.iter().enumerate().filter(|(_, e)| e.net == self.id).map(|(i, e)| (i, distance(&e.bits, &bits))).min_by_key(|&(_, d)| d);
            let at = match near {
                Some((i, d)) if d < feats.len() => {
                    cache[i].step(self, bits);
                    i
                }
                _ if cache.len() < ENTRIES => {
                    cache.push(self.refresh(bits));
                    cache.len() - 1
                }
                _ => {
                    let i = (0..ENTRIES).min_by_key(|&i| cache[i].used).unwrap();
                    cache[i] = self.refresh(bits);
                    i
                }
            };
            cache[at].used = *clock;
            (self.hidden(&cache[at].acc), cache[at].psqt)
        })
    }

    fn hidden(&self, acc: &[i32]) -> Vec<i8> {
        let (a, b) = acc.split_at(self.l1 / 2);
        a.iter().zip(b).map(|(&x, &y)| ((x.clamp(0, 255) * y.clamp(0, 255)) >> 9) as i8).collect()
    }

    fn value(&self, h: &[i8], stack: usize, psqt: i32) -> f32 {
        let half = self.l1 / 2;
        let c0 = affine::<L2>(&self.fc0[stack * L2 * half..][..L2 * half], &self.fc0_bias[stack * L2..][..L2], h);
        let x0 = squash(&c0, 7);
        let c1 = affine::<L2>(&self.fc1[stack * L2 * 2 * L2..][..L2 * 2 * L2], &self.fc1_bias[stack * L2..][..L2], &x0);
        let x1 = squash(&c1, 6);
        let w = &self.out[stack * 4 * L2..][..4 * L2];
        let o = self.out_bias[stack] + dot(&w[..2 * L2], &x0) + dot(&w[2 * L2..], &x1);
        (o + c0[L2 - 2] - c0[L2 - 1] + psqt) as f32 / OUT
    }
}

const ENTRIES: usize = 4;
static NETS: AtomicU64 = AtomicU64::new(0);

const WORDS: usize = FEATURES.div_ceil(64);
type Bits = [u64; WORDS];

struct Accumulator {
    net: u64,
    used: u64,
    bits: Bits,
    acc: Vec<i32>,
    psqt: [i32; STACKS],
}

impl Accumulator {
    fn add(&mut self, net: &Nnue, f: usize, sign: i32) {
        for (x, &w) in self.acc.iter_mut().zip(&net.ft[f * net.l1..][..net.l1]) {
            *x += sign * w as i32;
        }
        for (p, &w) in self.psqt.iter_mut().zip(&net.psqt[f * STACKS..][..STACKS]) {
            *p += sign * w;
        }
    }

    fn step(&mut self, net: &Nnue, bits: Bits) {
        for w in 0..WORDS {
            set(self.bits[w] ^ bits[w]).for_each(|b| self.add(net, w * 64 + b, if bits[w] >> b & 1 == 1 { 1 } else { -1 }));
        }
        self.bits = bits;
    }
}

fn set(x: u64) -> impl Iterator<Item = usize> {
    warchest::board::bits(x)
}

fn distance(a: &Bits, b: &Bits) -> usize {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones() as usize).sum()
}

thread_local! {
    static CACHE: RefCell<(u64, Vec<Accumulator>)> = const { RefCell::new((0, Vec::new())) };
}

fn dot(w: &[i8], x: &[i8]) -> i32 {
    w.iter().zip(x).map(|(&w, &x)| w as i32 * x as i32).sum()
}

fn affine<const N: usize>(w: &[i8], bias: &[i32], x: &[i8]) -> [i32; N] {
    std::array::from_fn(|o| bias[o] + dot(&w[o * x.len()..][..x.len()], x))
}

fn squash(c: &[i32; L2], shift: u32) -> [i8; 2 * L2] {
    let mut out = [0i8; 2 * L2];
    let (sqr, lin) = out.split_at_mut(L2);
    for ((s, l), &x) in sqr.iter_mut().zip(lin.iter_mut()).zip(c) {
        let x = x.clamp(-32767, 32767);
        *s = ((x * x) >> (2 * shift + 7)).min(127) as i8;
        *l = (x >> shift).clamp(0, 127) as i8;
    }
    out
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
        let half = self.l1 / 2;
        for i in 0..q.len() {
            let (row, desc) = q.row(i);
            let row = row.try_into().unwrap();
            let stack = self.stack(row);
            let (h, psqt) = self.accumulate(&features(row, desc));
            answer.value.push((self.value(&h, stack, psqt[stack]) / 2.0).tanh());
            let z: Vec<i32> = (0..self.r).map(|k| self.proj_bias[k] + dot(&self.proj[k * half..][..half], &h)).collect();
            let (start, mut zm) = (answer.prior.len(), z.clone());
            for d in desc.chunks_exact(ACT_BYTES) {
                let moves = move_features(row, d);
                zm.copy_from_slice(&z);
                for &f in moves.iter() {
                    zm.iter_mut().zip(&self.emb[f as usize * self.r..][..self.r]).for_each(|(a, &e)| *a += e);
                }
                let logit: i32 = zm.iter().zip(&self.head).map(|(&a, &w)| (a >> 7).clamp(0, 127) * w as i32).sum();
                let bias: i32 = moves.iter().map(|&f| self.move_bias[f as usize]).sum();
                answer.prior.push((logit + bias) as f32 / OUT);
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

    fn net(l1: usize, r: usize, scale: f32) -> Nnue {
        let mut rng = Rng::new(7);
        let mut floats = vec![l1 as f32, r as f32, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0];
        floats.extend((0..sizes(l1, r).iter().sum::<usize>()).map(|_| scale * (rng.unit_f64() as f32 - 0.5)));
        Nnue::from_floats(floats).unwrap()
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
            let mut mirrored = desc.to_vec();
            mirrored.chunks_exact_mut(ACT_BYTES).for_each(mirror_desc);
            let feats = sorted(&features(row, desc));
            assert!(feats.iter().all(|&f| (f as usize) < FEATURES) && feats.windows(2).all(|w| w[0] < w[1]), "{feats:?}");
            assert_eq!(feats, sorted(&features(&flipped, &mirrored)));
            assert_eq!(phase(row), phase(&flipped));
            most = most.max(feats.len());
            for d in desc.chunks_exact(ACT_BYTES) {
                let mut m = d.to_vec();
                mirror_desc(&mut m);
                assert!(move_features(row, d).iter().all(|&f| (f as usize) < MOVES));
                assert_eq!(sorted(&move_features(row, d)), sorted(&move_features(&flipped, &m)));
            }
        }
        assert!(most < 256, "{most} features");
    }

    #[test]
    fn a_zero_net_is_the_flat_oracle() {
        let q = positions();
        let answer = net(64, 16, 0.0).eval(&q);
        assert!(answer.value.iter().all(|&v| v == 0.0));
        assert_eq!(answer.prior, uniform(&q));
    }

    #[test]
    fn the_accumulator_cache_gives_what_a_refresh_gives() {
        let (q, nnue) = (positions(), net(64, 16, 0.5));
        for i in 0..q.len() {
            let row: &Row = q.row(i).0.try_into().unwrap();
            let feats = features(row, q.row(i).1);
            let (h, psqt) = nnue.accumulate(&feats);
            let mut bits = [0u64; WORDS];
            feats.iter().for_each(|&f| bits[f as usize / 64] |= 1 << (f % 64));
            let fresh = nnue.refresh(bits);
            assert_eq!((h, psqt), (nnue.hidden(&fresh.acc), fresh.psqt));
        }
    }
}
