use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, LazyLock};
use std::thread::JoinHandle;
use std::time::Instant;

use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::CudaBlas;
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DriverError,
    LaunchArgs, LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use half::f16;
use parking_lot::{Condvar, Mutex};

use crate::board::{board, NONE, N_HEXES};
use crate::net::{ln_block, Flat, Net, NetLayout, Span, BINS, BLOCKS, C, D, HIDDEN, LN_ACT, LN_TRUNK, TYPE};
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

const KERNELS: &str = concat!(include_str!("kernels.cu"), include_str!("trunk.cu"));

pub const PIPELINE: usize = 5;
const TILE: usize = 2048;
const ACT_TILE: usize = 64 * TILE;
const MERGE: usize = 1024;
const THREADS: u32 = 256;

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
    bias,
    trunk,
    value,
    actions,
    prior,
}

trait LaunchUnit {
    unsafe fn launch_unit(&mut self, cfg: LaunchConfig) -> Result<(), DriverError>;
}

impl LaunchUnit for LaunchArgs<'_> {
    unsafe fn launch_unit(&mut self, cfg: LaunchConfig) -> Result<(), DriverError> {
        self.launch(cfg).map(|_| ())
    }
}

fn blocks(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.max(1) as u32, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn warps(rows: usize) -> LaunchConfig {
    const WARPS: u32 = 8;
    LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(WARPS).max(1), 1, 1),
        block_dim: (32, WARPS, 1),
        shared_mem_bytes: 0,
    }
}

fn pack_mma(w: &[f32], base: usize, k: usize, n: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(k * n / 2);
    for kt in 0..k / 16 {
        for nt in 0..n / 8 {
            for lane in 0..32 {
                let (g, t) = (lane / 4, lane % 4);
                let at = |i: usize| f16::from_f32(w[base + (16 * kt + i) * n + 8 * nt + g]).to_bits();
                let pair = |i: usize| at(i) as u32 | ((at(i + 1) as u32) << 16);
                out.extend([pair(2 * t), pair(2 * t + 8)]);
            }
        }
    }
    out
}

fn pack_mma_a(w: &[f32], base: usize, k: usize, m: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(k * m / 2);
    for mt in 0..m / 16 {
        for kt in 0..k / 16 {
            for lane in 0..32 {
                let (g, t) = (lane / 4, lane % 4);
                let at = |r: usize, c: usize| f16::from_f32(w[base + (16 * kt + c) * m + 16 * mt + r]).to_bits();
                let pair = |r: usize, c: usize| at(r, c) as u32 | ((at(r, c + 1) as u32) << 16);
                out.extend([pair(g, 2 * t), pair(g + 8, 2 * t), pair(g, 2 * t + 8), pair(g + 8, 2 * t + 8)]);
            }
        }
    }
    out
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Stem {
    cards: u64,
    ids: u64,
    w_pile: u64,
    seat: u64,
    w_tok: u64,
    w_hex: u64,
    b_hex: u64,
    pos: u64,
    w_glob: u64,
}

unsafe impl cudarc::driver::DeviceRepr for Stem {}

const TRUNK_BOARDS: usize = 3;
const TRUNK_WARPS: usize = 12;
const TRUNK_THREADS: u32 = 32 * TRUNK_WARPS as u32;
const TRUNK_PITCH: usize = N_HEXES.next_multiple_of(16);
const TRUNK_SHARED: u32 = {
    let rows = (TRUNK_BOARDS * N_HEXES).next_multiple_of(16);
    let stem = 2 * (TRUNK_BOARDS * NTYPE * (TYPE + C) + 3 * TRUNK_BOARDS * C + TRUNK_BOARDS * LOOSE);
    let pitched = TRUNK_BOARDS * TRUNK_PITCH * (C + 8);
    let buf = if pitched > rows * (C + 8) + stem { pitched } else { rows * (C + 8) + stem };
    (buf * 2 + TRUNK_BOARDS * (2 * C + 8) * 2 + TRUNK_BOARDS * C * 4 + rows * 12) as u32
};
const HEX_K: usize = HEX_FACTS.next_multiple_of(16);

fn adjacency() -> Vec<u32> {
    let rows = (TRUNK_BOARDS * N_HEXES).next_multiple_of(16);
    let pitched = TRUNK_BOARDS * TRUNK_PITCH;
    let mut adj = vec![0.0f32; pitched * rows];
    for b in 0..TRUNK_BOARDS {
        for (hex, near) in board().neighbors.iter().enumerate() {
            for &n in near.iter().filter(|&&n| n != NONE) {
                adj[(b * TRUNK_PITCH + n as usize) * rows + b * N_HEXES + hex] = 1.0;
            }
        }
    }
    pack_mma_a(&adj, 0, pitched, rows)
}

struct Weights {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
    ln: CudaSlice<f32>,
    cards: CudaSlice<f32>,
    trunk_w: CudaSlice<u32>,
    trunk_g: CudaSlice<u32>,
    trunk_b: CudaSlice<f32>,
    trunk_ln: CudaSlice<f32>,
    trunk_hex: CudaSlice<u32>,
}

impl Weights {
    fn new(stream: &Arc<CudaStream>, l: &NetLayout, net: &Net) -> Res<Weights> {
        let flat: &Flat = net.flat();
        let (mut tw, mut tg, mut tb, mut tln) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, blk) in l.blocks.iter().enumerate() {
            let mut mix = vec![0.0f32; C * 2 * C];
            for k in 0..C {
                mix[k * 2 * C..k * 2 * C + C].copy_from_slice(&flat.w[blk.mix.w + (C + k) * C..][..C]);
                mix[k * 2 * C + C..(k + 1) * 2 * C].copy_from_slice(&flat.w[blk.mix.w + k * C..][..C]);
            }
            tw.extend(pack_mma(&mix, 0, C, 2 * C));
            tw.extend(pack_mma(&flat.w, blk.out.w, C, C));
            tg.extend(pack_mma_a(&flat.w, blk.pool.w, 2 * C, C));
            tb.extend((0..C).map(|j| flat.b[blk.mix.b + j] + flat.b[blk.pool.b + j]));
            tb.extend_from_slice(&flat.b[blk.out.b..blk.out.b + C]);
            for k in 0..2 {
                let n = l.norms[ln_block(i, k)];
                tln.extend_from_slice(&flat.ln[n.g..n.g + C]);
                tln.extend_from_slice(&flat.ln[n.b..n.b + C]);
            }
        }
        let n = l.norms[LN_TRUNK];
        tln.extend_from_slice(&flat.ln[n.g..n.g + C]);
        tln.extend_from_slice(&flat.ln[n.b..n.b + C]);
        let mut hex = vec![0.0f32; HEX_K * C];
        hex[..HEX_FACTS * C].copy_from_slice(&flat.w[l.hex_stem.w..l.hex_stem.w + HEX_FACTS * C]);
        let up = |x: &[f32]| stream.memcpy_stod(x).map_err(err);
        let upu = |x: &[u32]| stream.memcpy_stod(x).map_err(err);
        Ok(Weights {
            w: up(&flat.w)?,
            b: up(&flat.b)?,
            ln: up(&flat.ln)?,
            cards: up(&net.cards())?,
            trunk_w: upu(&tw)?,
            trunk_g: upu(&tg)?,
            trunk_b: up(&tb)?,
            trunk_ln: up(&tln)?,
            trunk_hex: upu(&pack_mma(&hex, 0, HEX_K, C))?,
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
    row_of: Wire<u32>,
    off: Wire<u32>,
    xpub: CudaSlice<f32>,
    input: CudaSlice<f32>,
    spatial: CudaSlice<f16>,
    tokens: CudaSlice<f16>,
    p: CudaSlice<f32>,
    hidden: CudaSlice<f32>,
    pooled: CudaSlice<f32>,
    logit: CudaSlice<f32>,
    out: CudaSlice<f32>,
    down: Host<f32>,
}

impl Scratch {
    fn new(s: &Arc<CudaStream>) -> Res<Scratch> {
        let zeros = |n: usize| s.alloc_zeros::<f32>(n).map_err(err);
        let halves = |n: usize| s.alloc_zeros::<f16>(n).map_err(err);
        Ok(Scratch {
            rows: Wire::new(s, TILE * ROW_BYTES)?,
            desc: Wire::new(s, ACT_TILE * ACT_BYTES)?,
            row_of: Wire::new(s, ACT_TILE)?,
            off: Wire::new(s, TILE + 1)?,
            xpub: zeros(TILE * PUBFEAT)?,
            input: zeros(TILE * (2 * C + LOOSE))?,
            spatial: halves(TILE * N_HEXES * C)?,
            tokens: halves(TILE * NTYPE * C)?,
            p: zeros(TILE * D)?,
            hidden: zeros(TILE * HIDDEN)?,
            pooled: zeros(TILE * C)?,
            logit: zeros(ACT_TILE)?,
            out: zeros(TILE + ACT_TILE)?,
            down: Host::new(s, TILE + ACT_TILE)?,
        })
    }
}

struct Card {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    k: Arc<Kernels>,
    layout: NetLayout,
    weights: Weights,
    adjacency: CudaSlice<u32>,
    card_facts: CudaSlice<f32>,
    locations: CudaSlice<u8>,
    s: Scratch,
}

fn lin(
    blas: &CudaBlas,
    stream: &Arc<CudaStream>,
    w: &CudaSlice<f32>,
    s: Span,
    inp: &CudaSlice<f32>,
    rows: usize,
    out: &mut CudaSlice<f32>,
) -> Res<()> {
    let (m, n, k) = (s.o as i32, rows as i32, s.i as i32);
    let (one, zero) = (1.0f32, 0.0f32);
    let weights = w.slice(s.w..s.w + s.i * s.o);
    let (a, _ka) = weights.device_ptr(stream);
    let (b, _kb) = inp.device_ptr(stream);
    let (c, _kc) = out.device_ptr_mut(stream);
    use cudarc::cublas::sys::{cublasComputeType_t, cublasGemmAlgo_t, cudaDataType_t::CUDA_R_32F};
    unsafe {
        cudarc::cublas::result::gemm_ex(
            *blas.handle(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m,
            n,
            k,
            (&one) as *const f32 as *const _,
            a as *const _,
            CUDA_R_32F,
            m,
            b as *const _,
            CUDA_R_32F,
            k,
            (&zero) as *const f32 as *const _,
            c as *mut _,
            CUDA_R_32F,
            m,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    }
    .map_err(err)
}

impl Card {
    fn new(module: &Module, net: &Net) -> Res<Card> {
        let stream = module.ctx.new_stream().map_err(err)?;
        let mut facts = vec![0.0; N_UNITS * CARD_FEATS];
        for u in 0..N_UNITS {
            write_card_features(u as u8, &mut facts[u * CARD_FEATS..(u + 1) * CARD_FEATS]);
        }
        let locations: Vec<u8> = board().is_location.iter().map(|&x| x as u8).collect();
        let layout = NetLayout::new();
        Ok(Card {
            blas: CudaBlas::new(stream.clone()).map_err(err)?,
            k: Arc::clone(&module.k),
            weights: Weights::new(&stream, &layout, net)?,
            adjacency: stream.memcpy_stod(&adjacency()).map_err(err)?,
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
        let Card { stream, blas, k, layout: l, weights: w, adjacency, card_facts, locations, s } = self;
        let n = r1 - r0;
        let (a0, a1) = (q.off[r0] as usize, q.off[r1] as usize);
        let na = a1 - a0;
        s.rows.put(stream, copy(&q.rows[r0 * ROW_BYTES..r1 * ROW_BYTES]))?;
        s.desc.put(stream, copy(&q.desc[a0 * ACT_BYTES..a1 * ACT_BYTES]))?;
        s.row_of.put(stream, |dst| {
            for r in 0..n {
                dst[q.actions(r0 + r).start - a0..q.actions(r0 + r).end - a0].fill(r as u32);
            }
            na
        })?;
        s.off.put(stream, |dst| {
            for (d, &o) in dst.iter_mut().zip(&q.off[r0..=r1]) {
                *d = o - a0 as u32;
            }
            n + 1
        })?;
        let (n_i, na_i) = (n as i32, na as i32);
        unsafe {
            stream
                .launch_builder(&k.expand_rows)
                .arg(&s.rows.dev)
                .arg(&*card_facts)
                .arg(&*locations)
                .arg(&mut s.xpub)
                .arg(&n_i)
                .launch_unit(blocks(n))
        }
        .map_err(err)?;
        let ptr = |x: &CudaSlice<f32>, at: usize| x.slice(at..).device_ptr(stream).0;
        let stem = Stem {
            cards: ptr(&w.cards, 0),
            ids: s.rows.dev.device_ptr(stream).0,
            w_pile: ptr(&w.w, l.pile.w),
            seat: ptr(&w.w, l.seat),
            w_tok: ptr(&w.w, l.tok_stem.w),
            w_hex: w.trunk_hex.device_ptr(stream).0,
            b_hex: ptr(&w.b, l.hex_stem.b),
            pos: ptr(&w.w, l.pos),
            w_glob: ptr(&w.w, l.glob_stem.w),
        };
        unsafe {
            stream
                .launch_builder(&k.trunk)
                .arg(&s.xpub)
                .arg(&stem)
                .arg(&w.trunk_w)
                .arg(&w.trunk_g)
                .arg(&w.trunk_b)
                .arg(&w.trunk_ln)
                .arg(&*adjacency)
                .arg(&mut s.input)
                .arg(&mut s.spatial)
                .arg(&mut s.tokens)
                .arg(&n_i)
                .launch_unit(LaunchConfig {
                    grid_dim: (n.div_ceil(TRUNK_BOARDS) as u32, 1, 1),
                    block_dim: (TRUNK_THREADS, 1, 1),
                    shared_mem_bytes: TRUNK_SHARED,
                })
        }
        .map_err(err)?;
        let bias = |span: Span, out: &mut CudaSlice<f32>| -> Res<()> {
            let (bv, width) = (w.b.slice(span.b..span.b + span.o), span.o as i32);
            unsafe { stream.launch_builder(&k.bias).arg(out).arg(&bv).arg(&n_i).arg(&width).launch_unit(warps(n)) }
                .map_err(err)
        };
        lin(blas, stream, &w.w, l.board_out, &s.input, n, &mut s.p)?;
        bias(l.board_out, &mut s.p)?;
        lin(blas, stream, &w.w, l.value_hidden, &s.p, n, &mut s.hidden)?;
        bias(l.value_hidden, &mut s.hidden)?;
        lin(blas, stream, &w.w, l.act_board, &s.p, n, &mut s.pooled)?;
        let (vw, vb) = (w.w.slice(l.value_out.w..), w.b.slice(l.value_out.b..));
        unsafe {
            stream.launch_builder(&k.value).arg(&s.hidden).arg(&vw).arg(&vb).arg(&mut s.out).arg(&n_i).launch_unit(warps(n))
        }
        .map_err(err)?;
        let norm = l.norms[LN_ACT];
        let (kind, role) = (w.w.slice(l.act_kind..), w.w.slice(l.act_role..));
        let (gamma, beta, w_out) = (w.ln.slice(norm.g..), w.ln.slice(norm.b..), w.w.slice(l.act_out.w..));
        unsafe {
            stream
                .launch_builder(&k.actions)
                .arg(&s.desc.dev)
                .arg(&s.row_of.dev)
                .arg(&s.tokens)
                .arg(&s.spatial)
                .arg(&kind)
                .arg(&role)
                .arg(&s.pooled)
                .arg(&gamma)
                .arg(&beta)
                .arg(&w_out)
                .arg(&mut s.logit)
                .arg(&na_i)
                .launch_unit(warps(na))
        }
        .map_err(err)?;
        let mut prior = s.out.slice_mut(n..);
        unsafe {
            stream.launch_builder(&k.prior).arg(&s.logit).arg(&s.off.dev).arg(&mut prior).arg(&n_i).launch_unit(warps(n))
        }
        .map_err(err)?;
        s.down.recv(stream, &s.out, n + na)
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
        BLOCKS,
        LOOSE,
        HIDDEN,
        BINS,
        ACT_BYTES,
    ];
    options.extend([
        format!("--gpu-architecture=compute_{major}{minor}"),
        format!("-DTRUNK_C={C}"),
        format!("-DTOKEN_W={TYPE}"),
        format!("-DTWARPS={TRUNK_WARPS}"),
        format!("-DHEX_K={HEX_K}"),
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
        k.trunk
            .set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                TRUNK_SHARED as i32,
            )
            .map_err(err)?;
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
    let torch_ctx = result::ctx::get_current().map_err(err)?.ok_or("PyTorch has no current CUDA context")?;
    let device = result::device::get(ordinal).map_err(err)?;
    let primary = unsafe { result::primary_ctx::retain(device) }.map_err(err)?;
    let same_context = primary == torch_ctx;
    unsafe { result::primary_ctx::release(device) }.map_err(err)?;
    if !same_context {
        return Err("PyTorch does not use the device primary CUDA context".into());
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
            .launch_unit(blocks(n))
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
    pub actions: AtomicU64,
    pub nanos: AtomicU64,
}

type Job = (Query, mpsc::Sender<Answer>);

pub struct Gpu {
    queues: Vec<Arc<Queue<Job>>>,
    lanes: Vec<Arc<Mutex<Card>>>,
    drivers: Vec<JoinHandle<()>>,
    next: AtomicUsize,
    pub stats: Arc<Stats>,
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
        let at = Instant::now();
        let (value, prior) = match lane.lock().eval(&merged) {
            Ok(out) => out,
            Err(e) => {
                eprintln!("cuda: {e}");
                return;
            }
        };
        stats.nanos.fetch_add(at.elapsed().as_nanos() as u64, Ordering::Relaxed);
        stats.calls.fetch_add(1, Ordering::Relaxed);
        stats.rows.fetch_add(value.len() as u64, Ordering::Relaxed);
        stats.actions.fetch_add(prior.len() as u64, Ordering::Relaxed);
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
        let stats = Arc::new(Stats::default());
        let (mut queues, mut lanes, mut drivers) = (Vec::new(), Vec::new(), Vec::new());
        for &o in ordinals {
            let module = Module::get(o)?;
            let queue = Arc::new(Queue::new());
            for p in 0..PIPELINE {
                let lane = Arc::new(Mutex::new(Card::new(&module, net)?));
                let (queue, card, stats) = (Arc::clone(&queue), Arc::clone(&lane), Arc::clone(&stats));
                drivers.push(
                    std::thread::Builder::new()
                        .name(format!("gpu-{o}.{p}"))
                        .spawn(move || drive(&queue, &card, &stats))
                        .map_err(err)?,
                );
                lanes.push(lane);
            }
            queues.push(queue);
        }
        Ok(Gpu { queues, lanes, drivers, next: AtomicUsize::new(0), stats })
    }

    pub fn set_weights(&self, net: &Net) -> Res<()> {
        for lane in &self.lanes {
            let mut card = lane.lock();
            card.stream.context().bind_to_thread().map_err(err)?;
            card.weights = Weights::new(&card.stream, &card.layout, net)?;
            card.stream.synchronize().map_err(err)?;
        }
        Ok(())
    }
}

impl Oracle for Gpu {
    fn eval(&self, q: Query) -> Answer {
        let (back, answer) = mpsc::channel();
        let g = self.next.fetch_add(1, Ordering::Relaxed) % self.queues.len();
        self.queues[g].push((q, back));
        answer.recv().expect("a gpu lane failed; its driver printed why")
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        for q in &self.queues {
            q.close();
        }
        for d in self.drivers.drain(..) {
            let _ = d.join();
        }
    }
}
