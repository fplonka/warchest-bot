use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, LazyLock};
use std::thread::JoinHandle;

use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DriverError,
    LaunchArgs, LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use half::f16;
use parking_lot::{Condvar, Mutex};

use crate::board::{board, N_HEXES};
use crate::memo::Memo;
use crate::net::{ln_block, Flat, Net, NetLayout, Span, BINS, BLOCKS, C, D, HIDDEN, LN_ACT, LN_TRUNK};
use crate::pbs::{
    HEX_BLOCK, HEX_CH, HEX_FACTS, LOOSE, MAX_COINS, NSLOT, NTYPE, OFF_CARDS, OFF_LOOSE, OFF_PILES, PILE_COUNTS,
    PLAYER_SCALARS, PUBFEAT, ROW_BAG_SIZE, ROW_BYTES, ROW_FD_SIZE, ROW_HAND_SIZE, ROW_HEX_HEIGHT, ROW_HEX_MARKER,
    ROW_HEX_OWNER, ROW_HEX_SLOT, ROW_IDS, ROW_INITIATIVE, ROW_INIT_MOVED, ROW_PILES, ROW_STACK_KIND, ROW_STACK_OWED,
    ROW_TO_ACT, ROW_WP,
};
use crate::search::{Answer, Oracle, Query, ACT_BYTES};
use crate::state::{CONT_CAP, PENDING_KINDS};
use crate::units::{write_card_features, CARD_FEATS, N_UNITS};

type Res<T> = Result<T, String>;

fn err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}

const KERNELS: &str = concat!(include_str!("kernels.cu"), include_str!("trunk.cu"), include_str!("heads.cu"));

const PIPELINE: usize = 5;
const TILE: usize = 2048;
const ACT_TILE: usize = 64 * TILE;
const MERGE: usize = 1024;

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
    dense,
    mix,
    out,
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

fn grid(x: usize, y: usize, threads: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (x.max(1) as u32, y as u32, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }
}

fn pack_mma(w: &[f32], k: usize, n: usize, tiles: usize) -> Vec<u32> {
    let column = |nt: usize, j: usize| {
        let i = nt % tiles;
        8 * (nt - i) + 32 * (i / 4) + 8 * (j / 2) + 2 * (i % 4) + j % 2
    };
    let mut out = Vec::with_capacity(k * n / 2);
    for kt in 0..k / 16 {
        for nt in 0..n / 8 {
            for lane in 0..32 {
                let (g, t) = (lane / 4, lane % 4);
                let at = |i: usize| f16::from_f32(w[(16 * kt + i) * n + column(nt, g)]).to_bits();
                let pair = |i: usize| at(i) as u32 | ((at(i + 1) as u32) << 16);
                out.extend([pair(2 * t), pair(2 * t + 8)]);
            }
        }
    }
    out
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Stem {
    tok: u64,
    ids: u64,
    w_hex: u64,
    b_hex: u64,
    pos: u64,
    w_glob: u64,
    gamma: u64,
    beta: u64,
}

unsafe impl cudarc::driver::DeviceRepr for Stem {}

#[repr(C)]
#[derive(Clone, Copy)]
struct Head {
    w_value: u64,
    b_value: u64,
    kind: u64,
    role: u64,
    gamma: u64,
    beta: u64,
    w_logit: u64,
}

unsafe impl cudarc::driver::DeviceRepr for Head {}

const KC: usize = 32;
const TRUNK_BOARDS: usize = 3;
const TRUNK_WARPS: usize = 8;
const DENSE_ROWS: usize = 32;
const DENSE_COLS: usize = 128;
const DENSE_WARPS: usize = 4;
const HEAD_ROWS: usize = 16;
const TRUNK_TILES: usize = C / 8 / TRUNK_WARPS;
const DENSE_TILES: usize = DENSE_COLS / 8 / DENSE_WARPS;
const POOLED: usize = 2 * C + LOOSE;
const HK: usize = POOLED.next_multiple_of(KC);
const BLOCK_PACK: usize = 5 * C * C / 2;

struct Weights {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
    ln: CudaSlice<f32>,
    tok: CudaSlice<f32>,
    trunk: CudaSlice<u32>,
    board: CudaSlice<u32>,
    mid: CudaSlice<u32>,
    mid_b: CudaSlice<f32>,
}

impl Weights {
    fn new(stream: &Arc<CudaStream>, l: &NetLayout, net: &Net) -> Res<Weights> {
        let flat: &Flat = net.flat();
        let mat = |s: Span| &flat.w[s.w..s.w + s.i * s.o];
        let mut trunk = Vec::with_capacity(BLOCKS * BLOCK_PACK);
        for blk in &l.blocks {
            trunk.extend(pack_mma(mat(blk.mix), 2 * C, C, TRUNK_TILES));
            trunk.extend(pack_mma(mat(blk.pool), 2 * C, C, DENSE_TILES));
            trunk.extend(pack_mma(mat(blk.out), C, C, TRUNK_TILES));
        }
        let mut board = vec![0.0f32; HK * D];
        board[..POOLED * D].copy_from_slice(mat(l.board_out));
        let mut mid = vec![0.0f32; D * D];
        for k in 0..D {
            mid[k * D..k * D + HIDDEN].copy_from_slice(&flat.w[l.value_hidden.w + k * HIDDEN..][..HIDDEN]);
            mid[k * D + HIDDEN..(k + 1) * D].copy_from_slice(&flat.w[l.act_board.w + k * C..][..C]);
        }
        let mut mid_b = vec![0.0f32; D];
        mid_b[..HIDDEN].copy_from_slice(&flat.b[l.value_hidden.b..l.value_hidden.b + HIDDEN]);
        let up = |x: &[f32]| stream.memcpy_stod(x).map_err(err);
        let upu = |x: &[u32]| stream.memcpy_stod(x).map_err(err);
        Ok(Weights {
            w: up(&flat.w)?,
            b: up(&flat.b)?,
            ln: up(&flat.ln)?,
            tok: up(&net.tokens())?,
            trunk: upu(&trunk)?,
            board: upu(&pack_mma(&board, HK, D, DENSE_TILES))?,
            mid: upu(&pack_mma(&mid, D, D, DENSE_TILES))?,
            mid_b: up(&mid_b)?,
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

    fn fill(&mut self, f: impl FnOnce(&mut [T]) -> usize) -> Res<()> {
        self.sent.synchronize().map_err(err)?;
        self.len = f(self.buf.as_mut_slice().map_err(err)?);
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

    fn put(&mut self, stream: &Arc<CudaStream>, f: impl FnOnce(&mut [T]) -> usize) -> Res<()> {
        self.host.fill(f)?;
        self.host.send(stream, &mut self.dev)
    }
}

fn copy<T: Copy>(src: &[T]) -> impl FnOnce(&mut [T]) -> usize + '_ {
    move |dst: &mut [T]| {
        dst[..src.len()].copy_from_slice(src);
        src.len()
    }
}

struct Scratch {
    rows: Wire<u8>,
    desc: Wire<u8>,
    off: Wire<u32>,
    xpub: CudaSlice<f32>,
    x: CudaSlice<f16>,
    a: CudaSlice<f16>,
    h: CudaSlice<f16>,
    pooled: CudaSlice<f16>,
    g: CudaSlice<f16>,
    board: CudaSlice<f16>,
    mid: CudaSlice<f16>,
    tokens: CudaSlice<f16>,
    out: CudaSlice<f32>,
    down: Host<f32>,
}

impl Scratch {
    fn new(s: &Arc<CudaStream>) -> Res<Scratch> {
        let halves = |n: usize| s.alloc_zeros::<f16>(n).map_err(err);
        let rows = TILE * N_HEXES;
        Ok(Scratch {
            rows: Wire::new(s, TILE * ROW_BYTES)?,
            desc: Wire::new(s, ACT_TILE * ACT_BYTES)?,
            off: Wire::new(s, TILE + 1)?,
            xpub: s.alloc_zeros(TILE * PUBFEAT).map_err(err)?,
            x: halves(rows * C)?,
            a: halves(rows * C)?,
            h: halves(rows * C)?,
            pooled: halves(TILE * HK)?,
            g: halves(TILE * C)?,
            board: halves(TILE * D)?,
            mid: halves(TILE * D)?,
            tokens: halves(TILE * NTYPE * C)?,
            out: s.alloc_zeros(TILE + ACT_TILE).map_err(err)?,
            down: Host::new(s, TILE + ACT_TILE)?,
        })
    }
}

struct Card {
    stream: Arc<CudaStream>,
    k: Arc<Kernels>,
    layout: NetLayout,
    weights: Weights,
    near: CudaSlice<u8>,
    card_facts: CudaSlice<f32>,
    locations: CudaSlice<u8>,
    s: Scratch,
}

macro_rules! launch {
    ($stream:expr, $kernel:expr, $cfg:expr, $($arg:expr),* $(,)?) => {
        unsafe { $stream.launch_builder(&$kernel)$(.arg($arg))*.launch_unit($cfg) }.map_err(err)?
    };
}

impl Card {
    fn new(module: &Module, net: &Net) -> Res<Card> {
        let stream = module.ctx.new_stream().map_err(err)?;
        let mut facts = vec![0.0; N_UNITS * CARD_FEATS];
        for u in 0..N_UNITS {
            write_card_features(u as u8, &mut facts[u * CARD_FEATS..(u + 1) * CARD_FEATS]);
        }
        let locations: Vec<u8> = board().is_location.iter().map(|&x| x as u8).collect();
        let near: Vec<u8> = board().neighbors.iter().flatten().copied().collect();
        let layout = NetLayout::new();
        Ok(Card {
            k: Arc::clone(&module.k),
            weights: Weights::new(&stream, &layout, net)?,
            near: stream.memcpy_stod(&near).map_err(err)?,
            card_facts: stream.memcpy_stod(&facts).map_err(err)?,
            locations: stream.memcpy_stod(&locations).map_err(err)?,
            s: Scratch::new(&stream)?,
            layout,
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
        let Card { stream, k, layout: l, weights: w, near, card_facts, locations, s } = self;
        let n = r1 - r0;
        let (a0, a1) = (q.off[r0] as usize, q.off[r1] as usize);
        s.rows.put(stream, copy(&q.rows[r0 * ROW_BYTES..r1 * ROW_BYTES]))?;
        s.desc.put(stream, copy(&q.desc[a0 * ACT_BYTES..a1 * ACT_BYTES]))?;
        s.off.put(stream, |dst| {
            for (d, &o) in dst.iter_mut().zip(&q.off[r0..=r1]) {
                *d = o - a0 as u32;
            }
            n + 1
        })?;
        let (n_i, rows) = (n as i32, (n * N_HEXES) as i32);
        let f32s = |x: &CudaSlice<f32>, at: usize| x.slice(at..).device_ptr(stream).0;
        let u32s = |at: usize| w.trunk.slice(at..).device_ptr(stream).0;
        let dense_grid = |cols: usize| grid(n.div_ceil(DENSE_ROWS), cols / DENSE_COLS, 32 * DENSE_WARPS as u32);
        let trunk_grid = grid(n.div_ceil(TRUNK_BOARDS), 1, 32 * TRUNK_WARPS as u32);
        let (c, c2, hk, d) = (C as i32, (2 * C) as i32, HK as i32, D as i32);
        let entry = |i: usize| l.norms[if i < BLOCKS { ln_block(i, 0) } else { LN_TRUNK }];
        launch!(stream, k.expand_rows, grid(n, 1, 256), &s.rows.dev, &*card_facts, &*locations, &mut s.xpub, &n_i);
        let stem = Stem {
            tok: f32s(&w.tok, 0),
            ids: s.rows.dev.device_ptr(stream).0,
            w_hex: f32s(&w.w, l.hex_stem.w),
            b_hex: f32s(&w.b, l.hex_stem.b),
            pos: f32s(&w.w, l.pos),
            w_glob: f32s(&w.w, l.glob_stem.w),
            gamma: f32s(&w.ln, entry(0).g),
            beta: f32s(&w.ln, entry(0).b),
        };
        launch!(stream, k.stem, grid(n, 1, C as u32), &s.xpub, &stem, &mut s.x, &mut s.a, &mut s.tokens, &mut s.pooled);
        for (i, blk) in l.blocks.iter().enumerate() {
            let (ln2, next) = (l.norms[ln_block(i, 1)], entry(i + 1));
            let (w1, wg, w2) = (u32s(i * BLOCK_PACK), u32s(i * BLOCK_PACK + C * C), u32s(i * BLOCK_PACK + 2 * C * C));
            let (g2, b2, gn, bn) = (f32s(&w.ln, ln2.g), f32s(&w.ln, ln2.b), f32s(&w.ln, next.g), f32s(&w.ln, next.b));
            let (bias_mix, bias_pool, bias_out) = (f32s(&w.b, blk.mix.b), f32s(&w.b, blk.pool.b), f32s(&w.b, blk.out.b));
            launch!(stream, k.dense, dense_grid(C), &s.pooled, &hk, &c2, &wg, &c, &bias_pool, &mut s.g, &n_i);
            launch!(stream, k.mix, trunk_grid, &s.a, &w1, &*near, &bias_mix, &s.g, &g2, &b2, &mut s.h, &rows);
            launch!(stream, k.out, trunk_grid, &s.h, &w2, &bias_out, &gn, &bn, &mut s.x, &mut s.a, &mut s.pooled, &rows);
        }
        let board_b = f32s(&w.b, l.board_out.b);
        launch!(stream, k.dense, dense_grid(D), &s.pooled, &hk, &hk, &w.board, &d, &board_b, &mut s.board, &n_i);
        launch!(stream, k.dense, dense_grid(D), &s.board, &d, &d, &w.mid, &d, &w.mid_b, &mut s.mid, &n_i);
        let norm = l.norms[LN_ACT];
        let head = Head {
            w_value: f32s(&w.w, l.value_out.w),
            b_value: f32s(&w.b, l.value_out.b),
            kind: f32s(&w.w, l.act_kind),
            role: f32s(&w.w, l.act_role),
            gamma: f32s(&w.ln, norm.g),
            beta: f32s(&w.ln, norm.b),
            w_logit: f32s(&w.w, l.act_out.w),
        };
        launch!(
            stream,
            k.heads,
            grid(n.div_ceil(HEAD_ROWS), 1, 256),
            &s.mid,
            &head,
            &s.desc.dev,
            &s.off.dev,
            &s.tokens,
            &s.a,
            &mut s.out,
            &n_i,
        );
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
        N_UNITS,
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
        BLOCKS,
        LOOSE,
        D,
        HIDDEN,
        BINS,
        ACT_BYTES,
    ];
    options.extend([
        format!("--gpu-architecture=compute_{major}{minor}"),
        format!("-DTRUNK_C={C}"),
        format!("-DKC={KC}"),
        format!("-DHK={HK}"),
        format!("-DTB={TRUNK_BOARDS}"),
        format!("-DTWARPS={TRUNK_WARPS}"),
        format!("-DDM={DENSE_ROWS}"),
        format!("-DDN={DENSE_COLS}"),
        format!("-DDWARPS={DENSE_WARPS}"),
        format!("-DHR={HEAD_ROWS}"),
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
        unsafe { ctx.disable_event_tracking() };
        let (major, minor) = ctx.compute_capability().map_err(err)?;
        let ptx = compile_ptx_with_opts(KERNELS, compile_options(major, minor)).map_err(|e| format!("nvrtc: {e:?}"))?;
        let k = Kernels::load(&ctx.load_module(ptx).map_err(err)?)?;
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
            card.weights = Weights::new(&card.stream, &card.layout, net)?;
            card.stream.synchronize().map_err(err)?;
        }
        self.memo.forget();
        Ok(())
    }
}

impl Oracle for Gpu {
    fn eval(&self, q: Query) -> Answer {
        self.memo.eval(q, |q| {
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
