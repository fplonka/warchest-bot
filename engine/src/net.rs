use std::ops::Range;

use crate::actions::N_KINDS;
use crate::board::{board, N_HEXES};
use crate::pbs::{HEX_FACTS, LOOSE, NSLOT, NTYPE, PILE_COUNTS};
use crate::search::ACT_BYTES;
use crate::selfplay::HORIZONS;
use crate::units::{write_card_features, CARD_FEATS, N_UNITS};

pub const D: usize = 256;
pub const LAYERS: usize = 8;
pub const HEADS: usize = 4;
pub const TOKENS: usize = N_HEXES + NTYPE + 1;
pub const STATE: usize = TOKENS - 1;
pub const FFN: usize = 2 * D;
const TYPE: usize = 64;
const SPAN: usize = 13;
const CLASSES: usize = 4;
const BIASES: usize = SPAN * SPAN + CLASSES * CLASSES;

pub fn attention_index() -> Vec<usize> {
    let coord = board().coord;
    let class = |t: usize| [N_HEXES, N_HEXES + NSLOT, STATE].iter().filter(|&&end| t >= end).count();
    let offset = |a: i8, b: i8| (a - b + (SPAN / 2) as i8) as usize;
    (0..TOKENS)
        .flat_map(|i| (0..TOKENS).map(move |j| (i, j)))
        .map(|(i, j)| match (i < N_HEXES, j < N_HEXES) {
            (true, true) => offset(coord[i].0, coord[j].0) * SPAN + offset(coord[i].1, coord[j].1),
            _ => SPAN * SPAN + class(i) * CLASSES + class(j),
        })
        .collect()
}

#[derive(Clone, Copy)]
pub struct Lin {
    pub w: usize,
    pub b: usize,
    pub i: usize,
    pub o: usize,
}

#[derive(Clone, Copy)]
pub struct Block {
    pub bias: usize,
    pub ln1: usize,
    pub qkv: Lin,
    pub proj: Lin,
    pub ln2: usize,
    pub up: Lin,
    pub down: Lin,
}

#[derive(Clone)]
pub struct Layout {
    pub none: usize,
    pub card1: Lin,
    pub card2: Lin,
    pub pile: usize,
    pub seat: usize,
    pub hex_stem: Lin,
    pub pos: usize,
    pub state_stem: Lin,
    pub blocks: [Block; LAYERS],
    pub ln: usize,
    pub value_hidden: Lin,
    pub value_out: Lin,
    pub short_out: Lin,
    pub kind: usize,
    pub move1: Lin,
    pub move2: Lin,
    pub len: usize,
}

struct Cursor(usize);

impl Cursor {
    fn take(&mut self, n: usize) -> usize {
        self.0 += n;
        self.0 - n
    }

    fn lin(&mut self, i: usize, o: usize) -> Lin {
        Lin { w: self.take(i * o), b: self.take(o), i, o }
    }
}

impl Layout {
    fn new() -> Layout {
        let mut c = Cursor(0);
        Layout {
            none: c.take(D),
            card1: c.lin(CARD_FEATS, TYPE),
            card2: c.lin(TYPE, D),
            pile: c.take(PILE_COUNTS * D),
            seat: c.take(2 * D),
            hex_stem: c.lin(HEX_FACTS, D),
            pos: c.take(N_HEXES * D),
            state_stem: c.lin(LOOSE, D),
            blocks: std::array::from_fn(|_| Block {
                bias: c.take(HEADS * BIASES),
                ln1: c.take(2 * D),
                qkv: c.lin(D, 3 * D),
                proj: c.lin(D, D),
                ln2: c.take(2 * D),
                up: c.lin(D, FFN),
                down: c.lin(FFN, D),
            }),
            ln: c.take(2 * D),
            value_hidden: c.lin(D, D),
            value_out: c.lin(D, 1),
            short_out: c.lin(D, HORIZONS.len() - 1),
            kind: c.take(N_KINDS * D),
            move1: c.lin((ACT_BYTES + 1) * D, D),
            move2: c.lin(D, 1),
            len: c.0,
        }
    }
}

pub struct Net {
    pub w: Vec<f32>,
    pub at: Layout,
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

impl Net {
    pub fn new(w: Vec<f32>) -> Result<Net, String> {
        let at = Layout::new();
        if w.len() != at.len {
            return Err(format!("{} weights do not match the network's {}", w.len(), at.len));
        }
        Ok(Net { w, at })
    }

    pub fn random(seed: u64) -> Net {
        let mut r = crate::rng::Rng::new(seed);
        let at = Layout::new();
        let mut w: Vec<f32> = (0..at.len).map(|_| (r.unit_f64() as f32 - 0.5) * 0.2).collect();
        for n in at.blocks.iter().flat_map(|b| [b.ln1, b.ln2]).chain([at.ln]) {
            w[n..n + D].fill(1.0);
        }
        Net { w, at }
    }

    pub fn load_bin(path: &str) -> std::io::Result<Net> {
        let raw = std::fs::read(path)?;
        let w = raw.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect();
        Net::new(w).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    pub fn get(&self, at: usize, n: usize) -> &[f32] {
        &self.w[at..at + n]
    }

    pub fn mat(&self, l: Lin) -> &[f32] {
        self.get(l.w, l.i * l.o)
    }

    fn apply(&self, l: Lin, x: &[f32]) -> Vec<f32> {
        self.mat(l).chunks_exact(l.i).zip(self.get(l.b, l.o)).map(|(row, b)| b + dot(row, x)).collect()
    }

    fn pick(&self, slot: usize, x: &[f32]) -> Vec<f32> {
        self.picks(slot..slot + 1).chunks_exact(D).map(|row| dot(row, x)).collect()
    }

    pub fn card_tokens(&self) -> Vec<f32> {
        (0..N_UNITS)
            .flat_map(|u| {
                let mut facts = [0.0f32; CARD_FEATS];
                write_card_features(u as u8, &mut facts);
                let hidden: Vec<f32> = self.apply(self.at.card1, &facts).into_iter().map(gelu).collect();
                self.apply(self.at.card2, &hidden)
            })
            .collect()
    }

    pub fn kind_tokens(&self) -> Vec<f32> {
        let b = self.get(self.at.move1.b, D);
        (0..N_KINDS).flat_map(|k| self.pick(0, self.get(self.at.kind + k * D, D)).into_iter().zip(b).map(|(x, b)| x + b)).collect()
    }

    pub fn none_tokens(&self) -> Vec<f32> {
        (1..ACT_BYTES).flat_map(|s| self.pick(s, self.get(self.at.none, D))).collect()
    }

    pub fn picks(&self, slots: Range<usize>) -> Vec<f32> {
        let m = self.at.move1;
        slots.flat_map(|s| self.mat(m).chunks_exact(m.i).flat_map(move |row| &row[s * D..][..D])).copied().collect()
    }

    pub fn state_head(&self) -> (Vec<f32>, Vec<f32>) {
        let v = self.at.value_hidden;
        ([self.picks(ACT_BYTES..ACT_BYTES + 1), self.mat(v).to_vec()].concat(), [vec![0.0; D], self.get(v.b, D).to_vec()].concat())
    }

    pub fn attention_bias(&self) -> Vec<f32> {
        let index = &attention_index();
        let biases = self.at.blocks.iter().flat_map(|b| (0..HEADS).map(move |h| b.bias + h * BIASES));
        biases.flat_map(|at| index.iter().map(move |&i| self.w[at + i])).collect()
    }
}
