use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, LazyLock};
use std::thread::JoinHandle;

use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DriverError, LaunchArgs,
    LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use half::f16;
use parking_lot::{Condvar, Mutex};

use crate::board::{board, N_HEXES};
use crate::memo::Memo;
use crate::net::{Layout, Lin, Net, D, FFN, HEADS, STATE, TOKENS};
use crate::pbs::{
    mirror_row, HEX_BLOCK, HEX_CH, HEX_FACTS, LOOSE, MAX_COINS, NSLOT, NTYPE, OFF_CARDS, OFF_LOOSE, OFF_PILES,
    PILE_COUNTS, PLAYER_SCALARS, PUBFEAT, ROW_BAG_SIZE, ROW_BYTES, ROW_FD_SIZE, ROW_HAND_SIZE, ROW_HEX_HEIGHT,
    ROW_HEX_MARKER, ROW_HEX_OWNER, ROW_HEX_SLOT, ROW_IDS, ROW_INITIATIVE, ROW_INIT_MOVED, ROW_PILES, ROW_STACK_KIND,
    ROW_STACK_OWED, ROW_TO_ACT, ROW_WP,
};
use crate::search::{mirror_desc, Answer, Oracle, Query, ACT_BYTES, COIN_PICKS, HEX_PICKS};
use crate::state::{CONT_CAP, PENDING_KINDS};
use crate::units::{write_card_features, CARD_FEATS, N_UNITS};

type Res<T> = Result<T, String>;

fn err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}

const KERNELS: &str = concat!(include_str!("kernels.cu"), include_str!("linear.cu"), include_str!("net.cu"));

const PIPELINE: usize = 5;
const TILE: usize = 2048;
const ACT_TILE: usize = 64 * TILE;
const MERGE: usize = 1024;
const HEAD_ROWS: usize = 16;
const BM: usize = 128;
const BN: usize = 128;
const WARPS_M: usize = 2;
const WARPS_N: usize = 4;
const BK: usize = 32;
const STAGES: usize = 3;
const LINEAR_SMEM: usize = STAGES * (BM + BN) * BK * 2;

macro_rules! kernels {
    ($($name:ident,)*) => {
        struct Kernels { $($name: CudaFunction,)* }

        impl Kernels {
            fn load(m: &Arc<CudaModule>) -> Res<Kernels> {
                $(let $name = {
                    let at = concat!("k_", stringify!($name));
                    m.load_function(at).map_err(|e| format!("kernel {at}: {e:?}"))?
                };)*
                Ok(Kernels { $($name,)* })
            }
        }
    };
}

kernels! {
    expand_rows,
    stem,
    linear,
    norm,
    attend,
    heads,
}

trait LaunchUnit {
    unsafe fn launch_unit(&mut self, cfg: LaunchConfig) -> Result<(), DriverError>;
}

impl LaunchUnit for LaunchArgs<'_> {
    unsafe fn launch_unit(&mut self, cfg: LaunchConfig) -> Result<(), DriverError> {
        self.launch(cfg).map(|_| ())
    }
}

fn grid(x: usize, y: usize, threads: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (x.max(1) as u32, y as u32, 1), block_dim: (threads as u32, 1, 1), shared_mem_bytes: 0 }
}

macro_rules! launch {
    ($stream:expr, $kernel:expr, $cfg:expr, $($arg:expr),* $(,)?) => {
        unsafe { $stream.launch_builder(&$kernel)$(.arg($arg))*.launch_unit($cfg) }.map_err(err)?
    };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Stem {
    card: u64,
    pile: u64,
    seat: u64,
    hex_w: u64,
    hex_b: u64,
    pos: u64,
    state_w: u64,
    state_b: u64,
}

unsafe impl cudarc::driver::DeviceRepr for Stem {}

#[repr(C)]
#[derive(Clone, Copy)]
struct Head {
    kind: u64,
    none: u64,
    value_w: u64,
    value_b: u64,
    move_w: u64,
    coins: u64,
    hexes: u64,
    state: u64,
}

unsafe impl cudarc::driver::DeviceRepr for Head {}

#[derive(Clone, Copy)]
enum Epilogue {
    Bias,
    Gelu,
    Add,
}

struct Linear {
    w: CudaSlice<f16>,
    bias: CudaSlice<f32>,
    epilogue: i32,
}

impl Linear {
    fn new(stream: &Arc<CudaStream>, w: &[f32], bias: &[f32], epilogue: Epilogue) -> Res<Linear> {
        let w: Vec<f16> = w.iter().map(|&x| f16::from_f32(x)).collect();
        Ok(Linear {
            w: stream.memcpy_stod(&w).map_err(err)?,
            bias: stream.memcpy_stod(bias).map_err(err)?,
            epilogue: epilogue as i32,
        })
    }

    fn run(&self, stream: &Arc<CudaStream>, k: &Kernels, x: u64, y: u64, rows: usize) -> Res<()> {
        let (n, inner) = (self.bias.len(), self.w.len() / self.bias.len());
        let cfg = LaunchConfig { shared_mem_bytes: LINEAR_SMEM as u32, ..grid(n / BN, rows.div_ceil(BM), 32 * WARPS_M * WARPS_N) };
        let (m, n, inner) = (rows as i32, n as i32, inner as i32);
        launch!(stream, k.linear, cfg, &x, &self.w, &self.bias, &y, &m, &n, &inner, &self.epilogue);
        Ok(())
    }
}

struct Weights {
    at: Layout,
    params: CudaSlice<f32>,
    card: CudaSlice<f32>,
    kind: CudaSlice<f32>,
    none: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    layers: Vec<[Linear; 4]>,
    coins: Linear,
    hexes: Linear,
    state: Linear,
}

impl Weights {
    fn new(stream: &Arc<CudaStream>, net: &Net) -> Res<Weights> {
        let up = |x: &[f32]| stream.memcpy_stod(x).map_err(err);
        let lin = |l: Lin, e| Linear::new(stream, net.mat(l), net.get(l.b, l.o), e);
        let picks = |slots: std::ops::Range<usize>| Linear::new(stream, &net.picks(slots.clone()), &vec![0.0; slots.len() * D], Epilogue::Bias);
        let layers = net.at.blocks.iter().map(|b| {
            Ok([lin(b.qkv, Epilogue::Bias)?, lin(b.proj, Epilogue::Add)?, lin(b.up, Epilogue::Gelu)?, lin(b.down, Epilogue::Add)?])
        });
        let (state_w, state_b) = net.state_head();
        Ok(Weights {
            at: net.at.clone(),
            params: up(&net.w)?,
            card: up(&net.card_tokens())?,
            kind: up(&net.kind_tokens())?,
            none: up(&net.none_tokens())?,
            bias: up(&net.attention_bias())?,
            layers: layers.collect::<Res<_>>()?,
            coins: picks(COIN_PICKS)?,
            hexes: picks(HEX_PICKS)?,
            state: Linear::new(stream, &state_w, &state_b, Epilogue::Bias)?,
        })
    }
}

struct Host<T> {
    buf: PinnedHostSlice<T>,
    len: usize,
    sent: CudaEvent,
}

impl<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy> Host<T> {
    fn new(stream: &Arc<CudaStream>, cap: usize) -> Res<Host<T>> {
        let ctx = stream.context();
        Ok(Host {
            buf: unsafe { ctx.alloc_pinned::<T>(cap) }.map_err(err)?,
            len: 0,
            sent: ctx.new_event(None).map_err(err)?,
        })
    }

    fn fill(&mut self, src: &[T]) -> Res<()> {
        self.sent.synchronize().map_err(err)?;
        self.buf.as_mut_slice().map_err(err)?[..src.len()].copy_from_slice(src);
        self.len = src.len();
        Ok(())
    }

    fn send(&mut self, stream: &Arc<CudaStream>, dst: &mut CudaSlice<T>) -> Res<()> {
        let src = &self.buf.as_slice().map_err(err)?[..self.len];
        stream.memcpy_htod(src, &mut dst.slice_mut(0..self.len)).map_err(err)?;
        self.sent.record(stream).map_err(err)
    }

    fn recv(&mut self, stream: &Arc<CudaStream>, src: &CudaSlice<T>, n: usize) -> Res<Vec<T>> {
        self.sent.synchronize().map_err(err)?;
        stream.memcpy_dtoh(&src.slice(0..n), &mut self.buf).map_err(err)?;
        self.sent.record(stream).map_err(err)?;
        self.sent.synchronize().map_err(err)?;
        Ok(self.buf.as_slice().map_err(err)?[..n].to_vec())
    }
}

struct Wire<T> {
    host: Host<T>,
    dev: CudaSlice<T>,
}

impl<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy> Wire<T> {
    fn new(stream: &Arc<CudaStream>, cap: usize) -> Res<Wire<T>> {
        Ok(Wire { host: Host::new(stream, cap)?, dev: stream.alloc_zeros(cap).map_err(err)? })
    }

    fn put(&mut self, stream: &Arc<CudaStream>, src: &[T]) -> Res<()> {
        self.host.fill(src)?;
        self.host.send(stream, &mut self.dev)
    }
}

struct Scratch {
    rows: Wire<u8>,
    desc: Wire<u8>,
    off: Wire<u32>,
    xpub: CudaSlice<f32>,
    x: CudaSlice<f16>,
    a: CudaSlice<f16>,
    wide: CudaSlice<f16>,
    out: CudaSlice<f32>,
    down: Host<f32>,
}

impl Scratch {
    fn new(s: &Arc<CudaStream>) -> Res<Scratch> {
        let halves = |n: usize| s.alloc_zeros::<f16>(TILE * TOKENS * n).map_err(err);
        Ok(Scratch {
            rows: Wire::new(s, TILE * ROW_BYTES)?,
            desc: Wire::new(s, ACT_TILE * ACT_BYTES)?,
            off: Wire::new(s, TILE + 1)?,
            xpub: s.alloc_zeros(TILE * PUBFEAT).map_err(err)?,
            x: halves(D)?,
            a: halves(D)?,
            wide: halves(FFN.max(3 * D))?,
            out: s.alloc_zeros(TILE + ACT_TILE).map_err(err)?,
            down: Host::new(s, TILE + ACT_TILE)?,
        })
    }
}

struct Card {
    stream: Arc<CudaStream>,
    k: Arc<Kernels>,
    weights: Weights,
    card_facts: CudaSlice<f32>,
    locations: CudaSlice<u8>,
    s: Scratch,
}

impl Card {
    fn new(module: &Module, net: &Net) -> Res<Card> {
        let stream = module.ctx.new_stream().map_err(err)?;
        let mut facts = vec![0.0; N_UNITS * CARD_FEATS];
        for u in 0..N_UNITS {
            write_card_features(u as u8, &mut facts[u * CARD_FEATS..(u + 1) * CARD_FEATS]);
        }
        let locations: Vec<u8> = board().is_location.iter().map(|&x| x as u8).collect();
        Ok(Card {
            k: Arc::clone(&module.k),
            weights: Weights::new(&stream, net)?,
            card_facts: stream.memcpy_stod(&facts).map_err(err)?,
            locations: stream.memcpy_stod(&locations).map_err(err)?,
            s: Scratch::new(&stream)?,
            stream,
        })
    }

    fn eval(&mut self, q: &Query) -> Res<(Vec<f32>, Vec<f32>)> {
        self.stream.context().bind_to_thread().map_err(err)?;
        let n = q.len();
        let (mut value, mut prior) = (Vec::with_capacity(n), Vec::with_capacity(q.off[n] as usize));
        let mut r0 = 0;
        while r0 < n {
            let mut r1 = r0 + 1;
            while r1 < n && r1 - r0 < TILE && (q.off[r1 + 1] - q.off[r0]) as usize <= ACT_TILE {
                r1 += 1;
            }
            assert!((q.off[r1] - q.off[r0]) as usize <= ACT_TILE, "one position offers more moves than a tile holds");
            let out = self.tile(q, r0, r1)?;
            value.extend_from_slice(&out[..r1 - r0]);
            prior.extend_from_slice(&out[r1 - r0..]);
            r0 = r1;
        }
        Ok((value, prior))
    }

    fn tile(&mut self, q: &Query, r0: usize, r1: usize) -> Res<Vec<f32>> {
        let Card { stream, k, weights: w, card_facts, locations, s } = self;
        let n = r1 - r0;
        let (a0, a1) = (q.off[r0] as usize, q.off[r1] as usize);
        s.rows.put(stream, &q.rows[r0 * ROW_BYTES..r1 * ROW_BYTES])?;
        s.desc.put(stream, &q.desc[a0 * ACT_BYTES..a1 * ACT_BYTES])?;
        let off: Vec<u32> = q.off[r0..=r1].iter().map(|&o| o - a0 as u32).collect();
        s.off.put(stream, &off)?;
        let (m, n_i) = (n * TOKENS, n as i32);
        let m_i = m as i32;
        let at = &w.at;
        let p = |x: &CudaSlice<f32>, at: usize| x.slice(at..).device_ptr(stream).0;
        launch!(stream, k.expand_rows, grid(n, 1, 256), &s.rows.dev, &*card_facts, &*locations, &mut s.xpub, &n_i);
        let stem = Stem {
            card: p(&w.card, 0),
            pile: p(&w.params, at.pile),
            seat: p(&w.params, at.seat),
            hex_w: p(&w.params, at.hex_stem.w),
            hex_b: p(&w.params, at.hex_stem.b),
            pos: p(&w.params, at.pos),
            state_w: p(&w.params, at.state_stem.w),
            state_b: p(&w.params, at.state_stem.b),
        };
        launch!(stream, k.stem, grid(n, 1, D), &s.xpub, &s.rows.dev, &stem, &mut s.x, &n_i);
        let h = |x: &CudaSlice<f16>, at: usize| x.slice(at..).device_ptr(stream).0;
        let (x, a, wide) = (h(&s.x, 0), h(&s.a, 0), h(&s.wide, 0));
        let norm = grid(m.div_ceil(8), 1, 256);
        for (l, (b, [qkv, proj, up, down])) in at.blocks.iter().zip(&w.layers).enumerate() {
            let bias = p(&w.bias, l * HEADS * TOKENS * TOKENS);
            launch!(stream, k.norm, norm, &s.x, &p(&w.params, b.ln1), &mut s.a, &m_i);
            qkv.run(stream, k, a, wide, m)?;
            launch!(stream, k.attend, grid(n, HEADS, 2 * TOKENS), &s.wide, &bias, &mut s.a, &n_i);
            proj.run(stream, k, a, x, m)?;
            launch!(stream, k.norm, norm, &s.x, &p(&w.params, b.ln2), &mut s.a, &m_i);
            up.run(stream, k, a, wide, m)?;
            down.run(stream, k, wide, x, m)?;
        }
        launch!(stream, k.norm, norm, &s.x, &p(&w.params, at.ln), &mut s.a, &m_i);
        let (hexes, coins) = (N_HEXES * n, NTYPE * n);
        let head = Head {
            kind: p(&w.kind, 0),
            none: p(&w.none, 0),
            value_w: p(&w.params, at.value_out.w),
            value_b: p(&w.params, at.value_out.b),
            move_w: p(&w.params, at.move2.w),
            hexes: wide,
            coins: h(&s.wide, hexes * HEX_PICKS.len() * D),
            state: h(&s.wide, (hexes * HEX_PICKS.len() + coins * COIN_PICKS.len()) * D),
        };
        w.hexes.run(stream, k, a, head.hexes, hexes)?;
        w.coins.run(stream, k, h(&s.a, hexes * D), head.coins, coins)?;
        w.state.run(stream, k, h(&s.a, (hexes + coins) * D), head.state, n)?;
        launch!(stream, k.heads, grid(n.div_ceil(HEAD_ROWS), 1, 256), &head, &s.desc.dev, &s.off.dev, &mut s.out, &n_i);
        s.down.recv(stream, &s.out, n + a1 - a0)
    }
}

struct Module {
    ctx: Arc<CudaContext>,
    k: Arc<Kernels>,
    torch: Arc<CudaStream>,
}

static MODULES: LazyLock<Mutex<HashMap<usize, Arc<Module>>>> = LazyLock::new(|| Mutex::new(HashMap::new()));

macro_rules! defines {
    ($($name:ident),* $(,)?) => {
        vec![$(format!("-D{}={}", stringify!($name), $name)),*]
    };
}

fn compile_options(major: i32, minor: i32) -> CompileOptions {
    let mut options = defines![
        ROW_BYTES,
        PUBFEAT,
        N_HEXES,
        HEX_CH,
        HEX_FACTS,
        HEX_BLOCK,
        NTYPE,
        NSLOT,
        PILE_COUNTS,
        CARD_FEATS,
        OFF_PILES,
        OFF_CARDS,
        OFF_LOOSE,
        PLAYER_SCALARS,
        ROW_IDS,
        ROW_HEX_OWNER,
        ROW_HEX_SLOT,
        ROW_HEX_HEIGHT,
        ROW_HEX_MARKER,
        ROW_PILES,
        ROW_HAND_SIZE,
        ROW_FD_SIZE,
        ROW_BAG_SIZE,
        ROW_INITIATIVE,
        ROW_INIT_MOVED,
        ROW_TO_ACT,
        ROW_WP,
        ROW_STACK_KIND,
        ROW_STACK_OWED,
        PENDING_KINDS,
        CONT_CAP,
        LOOSE,
        D,
        HEADS,
        TOKENS,
        STATE,
        ACT_BYTES,
    ];
    options.extend([
        format!("--gpu-architecture=compute_{major}{minor}"),
        format!("-DHR={HEAD_ROWS}"),
        format!("-DBM={BM}"),
        format!("-DBN={BN}"),
        format!("-DBK={BK}"),
        format!("-DWARPS_M={WARPS_M}"),
        format!("-DWARPS_N={WARPS_N}"),
        format!("-DSTAGES={STAGES}"),
        format!("-DMAX_COINS={MAX_COINS:.1}f"),
    ]);
    CompileOptions {
        options,
        include_paths: vec!["/usr/local/cuda/include".into(), "/usr/include".into()],
        ..Default::default()
    }
}

impl Module {
    fn get(ordinal: usize) -> Res<Arc<Module>> {
        let mut modules = MODULES.lock();
        if let Some(m) = modules.get(&ordinal) {
            m.ctx.bind_to_thread().map_err(err)?;
            return Ok(Arc::clone(m));
        }
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("device {ordinal}: {e:?}"))?;
        ctx.set_blocking_synchronize().map_err(err)?;
        unsafe { ctx.disable_event_tracking() };
        let (major, minor) = ctx.compute_capability().map_err(err)?;
        let ptx = compile_ptx_with_opts(KERNELS, compile_options(major, minor)).map_err(|e| format!("nvrtc: {e:?}"))?;
        let k = Kernels::load(&ctx.load_module(ptx).map_err(err)?)?;
        let smem = cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
        k.linear.set_attribute(smem, LINEAR_SMEM as i32).map_err(err)?;
        let torch = ctx.new_stream().map_err(err)?;
        let m = Arc::new(Module { ctx, k: Arc::new(k), torch });
        modules.insert(ordinal, Arc::clone(&m));
        Ok(m)
    }
}

pub fn expand_rows_torch(rows: u64, cards: u64, locations: u64, out: u64, n: usize, stream: usize, ordinal: i32) -> Res<()> {
    use cudarc::driver::{result, sys};
    if n == 0 {
        return Ok(());
    }
    let m = Module::get(ordinal as usize)?;
    let before = m.ctx.new_event(None).map_err(err)?;
    unsafe {
        result::event::record(before.cu_event(), stream as sys::CUstream).map_err(err)?;
    }
    m.torch.wait(&before).map_err(err)?;
    let rows = unsafe { m.torch.upgrade_device_ptr::<u8>(rows, n * ROW_BYTES) };
    let cards = unsafe { m.torch.upgrade_device_ptr::<f32>(cards, N_UNITS * CARD_FEATS) };
    let locations = unsafe { m.torch.upgrade_device_ptr::<u8>(locations, N_HEXES) };
    let mut out = unsafe { m.torch.upgrade_device_ptr::<f32>(out, n * PUBFEAT) };
    let n_i = n as i32;
    unsafe {
        m.torch
            .launch_builder(&m.k.expand_rows)
            .arg(&rows)
            .arg(&cards)
            .arg(&locations)
            .arg(&mut out)
            .arg(&n_i)
            .launch_unit(grid(n, 1, 256))
            .map_err(err)?;
    }
    rows.leak();
    cards.leak();
    locations.leak();
    out.leak();
    let after = m.torch.record_event(None).map_err(err)?;
    unsafe {
        result::stream::wait_event(stream as sys::CUstream, after.cu_event(), sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT)
            .map_err(err)
    }
}

struct Queue<T> {
    q: Mutex<(VecDeque<T>, bool)>,
    ready: Condvar,
}

impl<T> Queue<T> {
    fn new() -> Queue<T> {
        Queue { q: Mutex::new((VecDeque::new(), false)), ready: Condvar::new() }
    }

    fn push(&self, x: T) {
        self.q.lock().0.push_back(x);
        self.ready.notify_one();
    }

    fn take(&self, most: usize) -> Vec<T> {
        let mut g = self.q.lock();
        loop {
            if !g.0.is_empty() {
                let n = g.0.len().min(most);
                let out = g.0.drain(..n).collect();
                if !g.0.is_empty() {
                    self.ready.notify_one();
                }
                return out;
            }
            if g.1 {
                return Vec::new();
            }
            self.ready.wait(&mut g);
        }
    }

    fn close(&self) {
        self.q.lock().1 = true;
        self.ready.notify_all();
    }
}

#[derive(Default)]
pub struct Stats {
    pub calls: AtomicU64,
    pub rows: AtomicU64,
}

type Job = (Query, mpsc::Sender<Answer>);

pub struct Gpu {
    queue: Arc<Queue<Job>>,
    lanes: Vec<Arc<Mutex<Card>>>,
    drivers: Vec<JoinHandle<()>>,
    memo: Memo,
    pub stats: Vec<Arc<Stats>>,
}

fn drive(queue: &Queue<Job>, lane: &Mutex<Card>, stats: &Stats) {
    let mut merged = Query::default();
    loop {
        let jobs = queue.take(MERGE);
        if jobs.is_empty() {
            return;
        }
        merged.clear();
        for (q, _) in &jobs {
            merged.append(q);
        }
        let (value, prior) = match lane.lock().eval(&merged) {
            Ok(out) => out,
            Err(e) => {
                eprintln!("cuda: {e}");
                return;
            }
        };
        stats.calls.fetch_add(1, Ordering::Relaxed);
        stats.rows.fetch_add(value.len() as u64, Ordering::Relaxed);
        let (mut r, mut a) = (0, 0);
        for (q, back) in jobs {
            let (n, na) = (q.len(), q.desc.len() / ACT_BYTES);
            let _ = back.send(Answer { value: value[r..r + n].to_vec(), prior: prior[a..a + na].to_vec() });
            (r, a) = (r + n, a + na);
        }
    }
}

impl Gpu {
    pub fn new(ordinals: &[usize], net: &Net) -> Res<Gpu> {
        if ordinals.is_empty() {
            return Err("no cuda device ordinals given".into());
        }
        let queue = Arc::new(Queue::new());
        let (mut lanes, mut drivers, mut stats) = (Vec::new(), Vec::new(), Vec::new());
        for &o in ordinals {
            let module = Module::get(o)?;
            let device = Arc::new(Stats::default());
            for p in 0..PIPELINE {
                let lane = Arc::new(Mutex::new(Card::new(&module, net)?));
                let (queue, card, stats) = (Arc::clone(&queue), Arc::clone(&lane), Arc::clone(&device));
                drivers.push(
                    std::thread::Builder::new()
                        .name(format!("gpu-{o}.{p}"))
                        .spawn(move || drive(&queue, &card, &stats))
                        .map_err(err)?,
                );
                lanes.push(lane);
            }
            stats.push(device);
        }
        Ok(Gpu { queue, lanes, drivers, memo: Memo::default(), stats })
    }

    pub fn set_weights(&self, net: &Net) -> Res<()> {
        for lane in &self.lanes {
            let mut card = lane.lock();
            card.stream.context().bind_to_thread().map_err(err)?;
            card.weights = Weights::new(&card.stream, net)?;
            card.stream.synchronize().map_err(err)?;
        }
        self.memo.forget();
        Ok(())
    }
}

fn face_movers(q: &mut Query) {
    for r in 0..q.len() {
        let (at, acts) = (r * ROW_BYTES..(r + 1) * ROW_BYTES, q.actions(r));
        if q.rows[at.start + ROW_TO_ACT] == 1 {
            let src = <[u8; ROW_BYTES]>::try_from(&q.rows[at.clone()]).expect("a row is ROW_BYTES long");
            mirror_row(&src, &mut q.rows[at]);
            q.desc[acts.start * ACT_BYTES..acts.end * ACT_BYTES].chunks_exact_mut(ACT_BYTES).for_each(mirror_desc);
        }
    }
}

impl Oracle for Gpu {
    fn eval(&self, q: &Query) -> Answer {
        self.memo.eval(q, |mut q| {
            face_movers(&mut q);
            let (back, answer) = mpsc::channel();
            self.queue.push((q, back));
            answer.recv().expect("a gpu lane failed; its driver printed why")
        })
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.queue.close();
        for d in self.drivers.drain(..) {
            let _ = d.join();
        }
    }
}
