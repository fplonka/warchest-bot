use crate::actions::N_KINDS;
use crate::board::N_HEXES;
use crate::pbs::{HEX_FACTS, LOOSE, PILE_COUNTS};
use crate::units::{write_card_features, CARD_FEATS, N_UNITS};

pub const TYPE: usize = 64;
pub const C: usize = 192;
pub const BLOCKS: usize = 12;
pub const D: usize = 384;
pub const HIDDEN: usize = 192;
pub const BINS: usize = 16;

#[derive(Clone, Copy, Debug, Default)]
pub struct Span {
    pub w: usize,
    pub b: usize,
    pub i: usize,
    pub o: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NormSpan {
    pub g: usize,
    pub b: usize,
    pub width: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BlockSpan {
    pub mix: Span,
    pub pool: Span,
    pub out: Span,
}

#[derive(Clone, Debug)]
pub struct NetLayout {
    pub card: [Span; 2],
    pub pile: Span,
    pub seat: usize,
    pub hex_stem: Span,
    pub tok_stem: Span,
    pub pos: usize,
    pub glob_stem: Span,
    pub blocks: [BlockSpan; BLOCKS],
    pub board_out: Span,
    pub value_hidden: Span,
    pub value_out: Span,
    pub act_kind: usize,
    pub act_role: usize,
    pub act_board: Span,
    pub act_out: Span,
    pub norms: Vec<NormSpan>,
    pub w_len: usize,
    pub b_len: usize,
    pub ln_len: usize,
}

pub const fn ln_block(i: usize, half: usize) -> usize {
    2 * i + half
}
pub const LN_TRUNK: usize = 2 * BLOCKS;
pub const LN_ACT: usize = LN_TRUNK + 1;

#[derive(Default)]
struct Cursor {
    w: usize,
    b: usize,
    ln: usize,
}

impl Cursor {
    fn lin(&mut self, i: usize, o: usize, bias: bool) -> Span {
        let s = Span {
            w: self.w,
            b: if bias { self.b } else { usize::MAX },
            i,
            o,
        };
        self.w += i * o;
        if bias {
            self.b += o;
        }
        s
    }

    fn embed(&mut self, n: usize) -> usize {
        let at = self.w;
        self.w += n;
        at
    }

    fn norm(&mut self, width: usize) -> NormSpan {
        let s = NormSpan { g: self.ln, b: self.ln + width, width };
        self.ln += 2 * width;
        s
    }
}

impl Default for NetLayout {
    fn default() -> Self {
        Self::new()
    }
}

impl NetLayout {
    pub fn new() -> Self {
        let mut c = Cursor::default();
        let mut l = Self {
            card: [c.lin(CARD_FEATS, TYPE, true), c.lin(TYPE, TYPE, true)],
            pile: c.lin(PILE_COUNTS, TYPE, false),
            seat: c.embed(2 * TYPE),
            hex_stem: c.lin(HEX_FACTS, C, true),
            tok_stem: c.lin(TYPE, C, false),
            pos: c.embed(N_HEXES * C),
            glob_stem: c.lin(LOOSE, C, false),
            blocks: std::array::from_fn(|_| BlockSpan {
                mix: c.lin(2 * C, C, true),
                pool: c.lin(2 * C, C, true),
                out: c.lin(C, C, true),
            }),
            board_out: c.lin(2 * C + LOOSE, D, true),
            value_hidden: c.lin(D, HIDDEN, true),
            value_out: c.lin(HIDDEN, BINS, true),
            act_kind: c.embed(N_KINDS * C),
            act_role: c.embed(5 * C),
            act_board: c.lin(D, C, false),
            act_out: c.lin(C, 1, true),
            norms: (0..2 * BLOCKS + 2).map(|_| c.norm(C)).collect(),
            w_len: 0,
            b_len: 0,
            ln_len: 0,
        };
        (l.w_len, l.b_len, l.ln_len) = (c.w, c.b, c.ln);
        l
    }
}

#[derive(Clone, Default)]
pub struct Flat {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub ln: Vec<f32>,
}

#[derive(Clone, Default)]
pub struct Net {
    flat: Flat,
}

fn gelu(x: f32) -> f32 {
    let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

fn dense(flat: &Flat, s: Span, input: &[f32]) -> Vec<f32> {
    let mut out = if s.b == usize::MAX { vec![0.0; s.o] } else { flat.b[s.b..s.b + s.o].to_vec() };
    for (&x, weights) in input.iter().zip(flat.w[s.w..s.w + s.i * s.o].chunks_exact(s.o)) {
        for (d, &weight) in out.iter_mut().zip(weights) {
            *d = x.mul_add(weight, *d);
        }
    }
    out
}

impl Net {
    pub fn random(seed: u64) -> Net {
        let mut r = crate::rng::Rng::new(seed);
        let l = NetLayout::new();
        let mut draw = |n: usize| -> Vec<f32> { (0..n).map(|_| (r.unit_f64() as f32 - 0.5) * 0.2).collect() };
        let (w, b) = (draw(l.w_len), draw(l.b_len));
        let mut ln = vec![0.0; l.ln_len];
        for n in &l.norms {
            ln[n.g..n.g + n.width].fill(1.0);
        }
        Net::from_flat(&w, &b, &ln).expect("a random net matches the layout")
    }

    pub fn from_flat(w: &[f32], b: &[f32], ln: &[f32]) -> Result<Self, String> {
        let l = NetLayout::new();
        if (w.len(), b.len(), ln.len()) != (l.w_len, l.b_len, l.ln_len) {
            return Err(format!(
                "weight sizes {}/{}/{} do not match the network's {}/{}/{}",
                w.len(),
                b.len(),
                ln.len(),
                l.w_len,
                l.b_len,
                l.ln_len
            ));
        }
        Ok(Self {
            flat: Flat { w: w.to_vec(), b: b.to_vec(), ln: ln.to_vec() },
        })
    }

    pub fn load_bin(path: &str) -> std::io::Result<Self> {
        let raw = std::fs::read(path)?;
        let mut at = 0;
        let mut floats = || {
            let n = u32::from_le_bytes(raw[at..at + 4].try_into().unwrap()) as usize;
            let out: Vec<f32> = raw[at + 4..at + 4 + 4 * n]
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect();
            at += 4 + 4 * n;
            out
        };
        let (w, b, ln) = (floats(), floats(), floats());
        Self::from_flat(&w, &b, &ln).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn flat(&self) -> &Flat {
        &self.flat
    }

    pub fn tokens(&self) -> Vec<f32> {
        let l = NetLayout::new();
        let w = &self.flat.w;
        let cards = (0..N_UNITS).map(|u| {
            let mut facts = [0.0f32; CARD_FEATS];
            write_card_features(u as u8, &mut facts);
            let hidden: Vec<f32> = dense(&self.flat, l.card[0], &facts).into_iter().map(gelu).collect();
            dense(&self.flat, l.card[1], &hidden)
        });
        let seats = w[l.seat..l.seat + 2 * TYPE].chunks_exact(TYPE).map(<[f32]>::to_vec);
        let piles = w[l.pile.w..l.pile.w + PILE_COUNTS * TYPE].chunks_exact(TYPE).map(<[f32]>::to_vec);
        cards.chain(seats).chain(piles).flat_map(|t| dense(&self.flat, l.tok_stem, &t)).collect()
    }
}
