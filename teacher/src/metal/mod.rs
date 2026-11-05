use std::ffi::c_void;
use std::ptr::NonNull;

use half::f16;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use parking_lot::Mutex;

use crate::memo::Memo;
use crate::net::{Lin, Net, D, FFN, HEADS, STATE, TOKENS};
use engine::pbs::{
    HEX_CH, HEX_FACTS, LOOSE, MAX_COINS, NSLOT, NTYPE, PILE_COUNTS, PLAYER_SCALARS, ROW_BAG_SIZE, ROW_BYTES, ROW_FD_SIZE,
    ROW_HAND_SIZE, ROW_HEX_HEIGHT, ROW_HEX_MARKER, ROW_HEX_OWNER, ROW_HEX_SLOT, ROW_IDS, ROW_INITIATIVE, ROW_INIT_MOVED,
    ROW_PILES, ROW_STACK_KIND, ROW_STACK_OWED, ROW_TO_ACT, ROW_WP,
};
use engine::search::{Answer, Oracle, Query, ACT_BYTES, COIN_PICKS, HEX_PICKS};
use warchest::board::{board, N_HEXES};
use warchest::state::{CONT_CAP, PENDING_KINDS};

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

type Res<T> = Result<T, String>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Encoder = ProtocolObject<dyn MTLComputeCommandEncoder>;
type Device = ProtocolObject<dyn MTLDevice>;

const TILE: usize = 512;
const ACT_TILE: usize = 64 * TILE;
const HEAD_ROWS: usize = 16;
const PANEL_ROWS: usize = 16;
const SMALL: Tiling = Tiling { groups: 8, frags: 4 };
const LARGE: Tiling = Tiling { groups: 4, frags: 8 };
const LARGE_FROM: usize = 8 * TOKENS;

#[derive(Clone, Copy)]
struct Tiling {
    groups: usize,
    frags: usize,
}

macro_rules! defines {
    ($($name:ident),* $(,)?) => {
        [$(format!("#define {} {}\n", stringify!($name), $name)),*].concat()
    };
}

fn source() -> String {
    let defines = defines![
        D, HEADS, TOKENS, STATE, N_HEXES, NTYPE, NSLOT, HEX_CH, HEX_FACTS, PILE_COUNTS, LOOSE, PLAYER_SCALARS, CONT_CAP,
        PENDING_KINDS, ROW_BYTES, ROW_IDS, ROW_HEX_OWNER, ROW_HEX_SLOT, ROW_HEX_HEIGHT, ROW_HEX_MARKER, ROW_PILES,
        ROW_HAND_SIZE, ROW_FD_SIZE, ROW_BAG_SIZE, ROW_INITIATIVE, ROW_INIT_MOVED, ROW_TO_ACT, ROW_WP, ROW_STACK_KIND,
        ROW_STACK_OWED, ACT_BYTES, HEAD_ROWS,
    ];
    let panels = [("small", SMALL), ("large", LARGE)].map(|(name, t)| format!("PANEL({name}, {PANEL_ROWS}, {}, {})\n", t.groups, t.frags));
    let epilogues = [("GELU", Epilogue::Gelu), ("ADD", Epilogue::Add(0)), ("SPLIT", Epilogue::Split)].map(|(name, e)| format!("#define {name} {}\n", e.index()));
    format!("{defines}{}#define MAX_COINS {MAX_COINS:.1}f\n{}{}", epilogues.concat(), include_str!("net.metal"), panels.concat())
}

enum Arg<'a> {
    Buf(&'a Buffer, usize),
    Words(&'a [u32]),
}

fn dispatch(enc: &Encoder, p: &Pipeline, args: &[Arg], groups: [usize; 3], threads: usize) {
    enc.setComputePipelineState(p);
    for (i, arg) in args.iter().enumerate() {
        unsafe {
            match arg {
                Arg::Buf(b, at) => enc.setBuffer_offset_atIndex(Some(b), *at, i),
                Arg::Words(x) => enc.setBytes_length_atIndex(NonNull::from(*x).cast(), 4 * x.len(), i),
            }
        }
    }
    let [width, height, depth] = groups;
    enc.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width, height, depth }, MTLSize { width: threads, height: 1, depth: 1 });
}

#[derive(Clone, Copy)]
enum Epilogue {
    Store,
    Gelu,
    Add(usize),
    Split,
}

impl Epilogue {
    fn index(self) -> usize {
        match self {
            Epilogue::Store => 0,
            Epilogue::Gelu => 1,
            Epilogue::Add(_) => 2,
            Epilogue::Split => 3,
        }
    }
}

struct Kernels {
    stem: Pipeline,
    small: [Pipeline; 4],
    large: [Pipeline; 4],
    attend: Pipeline,
    heads: Pipeline,
}

impl Kernels {
    fn new(device: &Device) -> Res<Kernels> {
        let options = MTLCompileOptions::new();
        #[allow(deprecated)]
        options.setFastMathEnabled(true);
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(&source()), Some(&options))
            .map_err(|e| format!("metal compile: {e}"))?;
        let build = |name: &str, epilogue: i32| -> Res<Pipeline> {
            let values = MTLFunctionConstantValues::new();
            unsafe { values.setConstantValue_type_atIndex(NonNull::from(&epilogue).cast::<c_void>(), MTLDataType::Int, 0) };
            let function = library.newFunctionWithName_constantValues_error(&NSString::from_str(name), &values).map_err(|e| e.to_string())?;
            device.newComputePipelineStateWithFunction_error(&function).map_err(|e| e.to_string())
        };
        let epilogues = |name: &str| -> Res<[Pipeline; 4]> {
            let [store, gelu, add, split] = [Epilogue::Store, Epilogue::Gelu, Epilogue::Add(0), Epilogue::Split].map(|e| build(name, e.index() as i32));
            Ok([store?, gelu?, add?, split?])
        };
        Ok(Kernels {
            stem: build("stem", 0)?,
            small: epilogues("small")?,
            large: epilogues("large")?,
            attend: build("attend", 0)?,
            heads: build("heads", 0)?,
        })
    }
}

struct Linear {
    w: Buffer,
    bias: Buffer,
    n: usize,
    k: usize,
}

struct Layer {
    ln1: usize,
    qkv: Linear,
    proj: Linear,
    ln2: usize,
    up: Linear,
    down: Linear,
}

struct Weights {
    params: Buffer,
    pile: Buffer,
    hex: Buffer,
    loose: Buffer,
    card: Buffer,
    kind: Buffer,
    none: Buffer,
    bias: Buffer,
    locations: Buffer,
    layers: Vec<Layer>,
    hexes: Linear,
    coins: Linear,
    state: Linear,
    at: crate::net::Layout,
}

fn buffer(device: &Device, bytes: usize) -> Res<Buffer> {
    device.newBufferWithLength_options(bytes.max(16), MTLResourceOptions::StorageModeShared).ok_or_else(|| format!("no {bytes} byte buffer"))
}

fn fill<T: Copy>(b: &Buffer, data: &[T]) {
    assert!(std::mem::size_of_val(data) <= b.length());
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), b.contents().as_ptr().cast::<T>(), data.len()) };
}

fn upload<T: Copy>(device: &Device, data: &[T]) -> Res<Buffer> {
    let b = buffer(device, std::mem::size_of_val(data))?;
    fill(&b, data);
    Ok(b)
}

impl Linear {
    fn new(device: &Device, w: &[f32], bias: &[f32]) -> Res<Linear> {
        let (n, k) = (bias.len(), w.len() / bias.len());
        assert!([SMALL, LARGE].iter().all(|t| n % (8 * t.groups * t.frags) == 0) && k % D == 0, "a {n}x{k} linear does not tile");
        let swizzled: Vec<f16> = (0..k * n)
            .map(|i| {
                let (block, r, c) = (i / 64, i / 8 % 8, i % 8);
                let (kb, nb) = (block / (n / 8), block % (n / 8));
                f16::from_f32(w[(nb * 8 + c) * k + kb * 8 + r])
            })
            .collect();
        Ok(Linear { w: upload(device, &swizzled)?, bias: upload(device, bias)?, n, k })
    }
}

impl Weights {
    fn new(device: &Device, net: &Net) -> Res<Weights> {
        let lin = |l: Lin| Linear::new(device, net.mat(l), net.get(l.b, l.o));
        let picks = |slots: std::ops::Range<usize>| Linear::new(device, &net.picks(slots.clone()), &vec![0.0; slots.len() * D]);
        let layers = net.at.blocks.iter().map(|b| {
            Ok(Layer { ln1: b.ln1, qkv: lin(b.qkv)?, proj: lin(b.proj)?, ln2: b.ln2, up: lin(b.up)?, down: lin(b.down)? })
        });
        let (state_w, state_b) = net.state_head();
        let locations: Vec<u8> = board().is_location.iter().map(|&x| x as u8).collect();
        let columns = |at: usize, k: usize| -> Vec<f32> { (0..k * D).map(|i| net.w[at + i % D * k + i / D]).collect() };
        Ok(Weights {
            params: upload(device, &net.w)?,
            pile: upload(device, &columns(net.at.pile, PILE_COUNTS))?,
            hex: upload(device, &columns(net.at.hex_stem.w, HEX_FACTS))?,
            loose: upload(device, &columns(net.at.state_stem.w, LOOSE))?,
            card: upload(device, &net.card_tokens())?,
            kind: upload(device, &net.kind_tokens())?,
            none: upload(device, &net.none_tokens())?,
            bias: upload(device, &net.attention_bias())?,
            locations: upload(device, &locations)?,
            layers: layers.collect::<Res<_>>()?,
            hexes: picks(HEX_PICKS)?,
            coins: picks(COIN_PICKS)?,
            state: Linear::new(device, &state_w, &state_b)?,
            at: net.at.clone(),
        })
    }
}

struct Scratch {
    rows: Buffer,
    desc: Buffer,
    off: Buffer,
    x: Buffer,
    a: Buffer,
    wide: Buffer,
    out: Buffer,
}

impl Scratch {
    fn new(device: &Device) -> Res<Scratch> {
        let halves = |width: usize| buffer(device, TILE * TOKENS * width * 2);
        Ok(Scratch {
            rows: buffer(device, TILE * ROW_BYTES)?,
            desc: buffer(device, ACT_TILE * ACT_BYTES)?,
            off: buffer(device, (TILE + 1) * 4)?,
            x: halves(D)?,
            a: halves(D)?,
            wide: halves(FFN.max(3 * D))?,
            out: buffer(device, (TILE + ACT_TILE) * 4)?,
        })
    }
}

pub struct Metal {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    k: Kernels,
    w: Weights,
    s: Mutex<Scratch>,
    memo: Memo,
}

unsafe impl Send for Metal {}
unsafe impl Sync for Metal {}

impl Metal {
    pub fn new(net: &Net) -> Res<Metal> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no metal device")?;
        Ok(Metal {
            queue: device.newCommandQueue().ok_or("no command queue")?,
            k: Kernels::new(&device)?,
            w: Weights::new(&device, net)?,
            s: Mutex::new(Scratch::new(&device)?),
            memo: Memo::default(),
        })
    }

    fn run(&self, q: &Query) -> Res<Answer> {
        let s = self.s.lock();
        let n = q.len();
        let mut answer = Answer { value: Vec::with_capacity(n), prior: Vec::with_capacity(q.off[n] as usize) };
        for t in q.tiles(TILE, ACT_TILE) {
            self.tile(&s, q, t.start, t.end, &mut answer)?;
        }
        Ok(answer)
    }

    fn linear(&self, s: &Scratch, enc: &Encoder, l: &Linear, a: (&Buffer, usize), y: (&Buffer, usize), m: usize, epilogue: Epilogue) {
        let ln = match epilogue {
            Epilogue::Add(ln) => ln,
            _ => 0,
        };
        let args = [
            Arg::Buf(a.0, a.1 * 2),
            Arg::Buf(&l.w, 0),
            Arg::Buf(&l.bias, 0),
            Arg::Buf(y.0, y.1 * 2),
            Arg::Buf(&s.a, 0),
            Arg::Buf(&self.w.params, ln * 4),
            Arg::Words(&[m as u32, l.n as u32, l.k as u32]),
        ];
        let (pipelines, t) = if m < LARGE_FROM { (&self.k.small, SMALL) } else { (&self.k.large, LARGE) };
        dispatch(enc, &pipelines[epilogue.index()], &args, [l.n / (8 * t.groups * t.frags), m.div_ceil(PANEL_ROWS), 1], 32 * t.groups);
    }

    fn tile(&self, s: &Scratch, q: &Query, r0: usize, r1: usize, answer: &mut Answer) -> Res<()> {
        let (k, w) = (&self.k, &self.w);
        let n = r1 - r0;
        let (a0, a1) = (q.off[r0] as usize, q.off[r1] as usize);
        fill(&s.rows, &q.rows[r0 * ROW_BYTES..r1 * ROW_BYTES]);
        fill(&s.desc, &q.desc[a0 * ACT_BYTES..a1 * ACT_BYTES]);
        let off: Vec<u32> = q.off[r0..=r1].iter().map(|&o| o - a0 as u32).collect();
        fill(&s.off, &off);
        let command = self.queue.commandBuffer().ok_or("no command buffer")?;
        let enc = command.computeCommandEncoder().ok_or("no compute encoder")?;
        let (at, p) = (&w.at, &w.params);
        let f = |offset: usize| Arg::Buf(p, offset * 4);
        let stem = [
            Arg::Buf(&s.rows, 0),
            Arg::Buf(&w.card, 0),
            Arg::Buf(&w.pile, 0),
            f(at.seat),
            Arg::Buf(&w.hex, 0),
            f(at.hex_stem.b),
            f(at.pos),
            Arg::Buf(&w.loose, 0),
            f(at.state_stem.b),
            Arg::Buf(&w.locations, 0),
            Arg::Buf(&s.x, 0),
            Arg::Buf(&s.a, 0),
            f(w.layers[0].ln1),
            Arg::Words(&[n as u32]),
        ];
        dispatch(&enc, &k.stem, &stem, [n, TOKENS, 1], D);
        let m = n * TOKENS;
        for (i, l) in w.layers.iter().enumerate() {
            self.linear(s, &enc, &l.qkv, (&s.a, 0), (&s.wide, 0), m, Epilogue::Split);
            let bias = i * HEADS * TOKENS * TOKENS * 4;
            let attend = [Arg::Buf(&s.wide, 0), Arg::Buf(&w.bias, bias), Arg::Buf(&s.a, 0), Arg::Words(&[n as u32])];
            dispatch(&enc, &k.attend, &attend, [n, HEADS, TOKENS / 16], 32);
            self.linear(s, &enc, &l.proj, (&s.a, 0), (&s.x, 0), m, Epilogue::Add(l.ln2));
            self.linear(s, &enc, &l.up, (&s.a, 0), (&s.wide, 0), m, Epilogue::Gelu);
            let next = w.layers.get(i + 1).map_or(at.ln, |l| l.ln1);
            self.linear(s, &enc, &l.down, (&s.wide, 0), (&s.x, 0), m, Epilogue::Add(next));
        }
        let (hexes, coins) = (N_HEXES * n, NTYPE * n);
        let (coin_out, state_out) = (hexes * 3 * D, hexes * 3 * D + coins * 2 * D);
        self.linear(s, &enc, &w.hexes, (&s.a, 0), (&s.wide, 0), hexes, Epilogue::Store);
        self.linear(s, &enc, &w.coins, (&s.a, hexes * D), (&s.wide, coin_out), coins, Epilogue::Store);
        self.linear(s, &enc, &w.state, (&s.a, (hexes + coins) * D), (&s.wide, state_out), n, Epilogue::Store);
        let heads = [
            Arg::Buf(&s.wide, 0),
            Arg::Buf(&s.wide, coin_out * 2),
            Arg::Buf(&s.wide, state_out * 2),
            Arg::Buf(&w.kind, 0),
            Arg::Buf(&w.none, 0),
            f(at.value_out.w),
            f(at.value_out.b),
            f(at.move2.w),
            Arg::Buf(&s.desc, 0),
            Arg::Buf(&s.off, 0),
            Arg::Buf(&s.out, 0),
            Arg::Words(&[n as u32]),
        ];
        dispatch(&enc, &k.heads, &heads, [n.div_ceil(HEAD_ROWS), 1, 1], 256);
        enc.endEncoding();
        command.commit();
        while !matches!(command.status(), MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error) {
            std::hint::spin_loop();
        }
        if let Some(e) = command.error() {
            return Err(format!("metal: {e}"));
        }
        let out = unsafe { std::slice::from_raw_parts(s.out.contents().as_ptr().cast::<f32>(), n + a1 - a0) };
        answer.value.extend_from_slice(&out[..n]);
        answer.prior.extend_from_slice(&out[n..]);
        Ok(())
    }
}

impl Oracle for Metal {
    fn eval(&self, q: &Query) -> Answer {
        self.memo.eval(q, |mut q| {
            q.face_movers();
            self.run(&q).expect("the metal oracle failed")
        })
    }
}
