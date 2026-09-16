use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
use cudarc::cublas::CudaBlas;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    DriverError, LaunchArgs, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use half::f16;

use crate::board::{board, NONE, N_HEXES};
use crate::contract::{Call, Prime, Reply};
use crate::net::{
    ln_block, Net, NetLayout, NormSpan, Span, ATTN, BINS, BLOCKS, C, D, HEADS, LN_ACT, LN_ATTN,
    LN_CFG, LN_TRUNK, TYPE,
};
use crate::pbs::{
    CFEAT, HEX_BLOCK, HEX_CH, HEX_FACTS, LOOSE, MAX_COINS, NSLOT, NTYPE, OFF_CARDS, OFF_LOOSE,
    OFF_PILES, PILE_COUNTS, PLAYER_SCALARS, PUBFEAT, ROW_BAG_SIZE, ROW_BYTES, ROW_FD_SIZE,
    ROW_HAND_SIZE, ROW_HEX_HEIGHT, ROW_HEX_MARKER, ROW_HEX_OWNER, ROW_HEX_SLOT, ROW_IDS,
    ROW_INITIATIVE, ROW_INIT_MOVED, ROW_PILES, ROW_STACK_KIND, ROW_STACK_OWED, ROW_TO_ACT, ROW_WP,
};
use crate::search::{Cfg, Cfr, Ent};
use crate::state::{CONT_CAP, PENDING_KINDS};
use crate::units::{write_card_features, CARD_FEATS, N_UNITS};

mod slot;
use slot::{
    Arr, Solve, B_P, C_CUR, C_PRIOR, C_QVAL, C_SUM, C_VISITS, DESC, FIELDS, G_QUERY,
    R_REACH, R_VALS, Y_BOARD_OF, Y_COFF,
};

type Res<T> = Result<T, String>;

trait Elem: cudarc::driver::DeviceRepr {
    const KIND: cudarc::cublas::sys::cudaDataType_t;
    const HALF: bool;
}

impl Elem for f32 {
    const KIND: cudarc::cublas::sys::cudaDataType_t = cudarc::cublas::sys::cudaDataType_t::CUDA_R_32F;
    const HALF: bool = false;
}

impl Elem for f16 {
    const KIND: cudarc::cublas::sys::cudaDataType_t = cudarc::cublas::sys::cudaDataType_t::CUDA_R_16F;
    const HALF: bool = true;
}

const KERNELS: &str = concat!(include_str!("kernels.cu"), include_str!("trunk.cu"), include_str!("cfr.cu"));

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
    gelu,
    norm_ip,
    bias,
    scatter,
    finish,
    act_feats,
    act_add,
    prior,
    trunk,
    public_cache,
    attention,
    attention_residual,
    attention_store,
    cfg_slots,
    cfg_seat,
    sum_slots,
    belief,
    value,
    policy_inputs,
    policy_offset,
    reach,
    cfr,
}

trait LaunchUnit {
    unsafe fn launch_unit(&mut self, cfg: LaunchConfig) -> Result<(), DriverError>;
}

impl LaunchUnit for LaunchArgs<'_> {
    unsafe fn launch_unit(&mut self, cfg: LaunchConfig) -> Result<(), DriverError> {
        self.launch(cfg).map(|_| ())
    }
}

macro_rules! launch {
    ($card:expr, $kernel:ident, $n:expr, $($arg:expr),+ $(,)?) => {{
        let n = $n;
        unsafe {
            $card.stream
                .launch_builder(&$card.k.$kernel)
                $(.arg($arg))+
                .launch_unit(spread(n))
        }
        .map_err(err)
    }};
}

macro_rules! carved {
    ($($name:ident),* $(,)?) => { $( let $name = $name.buf.as_mut().expect("carved"); )* };
}

mod cfr;
use cfr::Batch;

const THREADS: u32 = 256;

fn spread(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(THREADS).max(1), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn warp_rows(rows: usize) -> LaunchConfig {
    const WARPS: u32 = 8;
    LaunchConfig {
        grid_dim: ((rows as u32).div_ceil(WARPS).max(1), 1, 1),
        block_dim: (32, WARPS, 1),
        shared_mem_bytes: 0,
    }
}

pub struct Device {
    cards: Vec<Card>,
    n_gpus: usize,
    slot_bytes: usize,
}

pub const PIPELINE: usize = 5;

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

fn halved(w: &[f32]) -> Vec<f16> {
    w.iter().map(|&v| f16::from_f32(v)).collect()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Stem {
    cards: u64,
    card_of_row: u64,
    w_pile: u64,
    seat: u64,
    w_tok: u64,
    w_hex: u64,
    b_hex: u64,
    pos: u64,
    w_glob: u64,
}

unsafe impl cudarc::driver::DeviceRepr for Stem {}

struct Trunk {
    w: Vec<u32>,
    g: Vec<u32>,
    b: Vec<f32>,
    ln: Vec<f32>,
}

const TRUNK_BOARDS: usize = 3;
const TRUNK_WARPS: usize = 12;
const TRUNK_THREADS: u32 = 32 * TRUNK_WARPS as u32;
const TRUNK_SHARED: u32 = {
    let rows = (TRUNK_BOARDS * N_HEXES).next_multiple_of(16);
    (2 * rows * (C + 8) * 2 + TRUNK_BOARDS * (2 * C + 8) * 2 + TRUNK_BOARDS * C * 4 + N_HEXES * 6 * 4)
        as u32
};

const HEX_K: usize = HEX_FACTS.next_multiple_of(16);

fn hex_pack(l: &NetLayout, flat: &crate::net::Flat) -> Vec<u32> {
    let mut w = vec![0.0f32; HEX_K * C];
    w[..HEX_FACTS * C].copy_from_slice(&flat.w[l.hex_stem.w..l.hex_stem.w + HEX_FACTS * C]);
    pack_mma(&w, 0, HEX_K, C)
}

fn trunk_pack(l: &NetLayout, flat: &crate::net::Flat) -> Trunk {
    let mut t = Trunk { w: Vec::new(), g: Vec::new(), b: Vec::new(), ln: Vec::new() };
    for (i, blk) in l.blocks.iter().enumerate() {
        let mut mix = vec![0.0f32; C * 2 * C];
        for k in 0..C {
            mix[k * 2 * C..k * 2 * C + C].copy_from_slice(&flat.w[blk.mix.w + (C + k) * C..][..C]);
            mix[k * 2 * C + C..(k + 1) * 2 * C].copy_from_slice(&flat.w[blk.mix.w + k * C..][..C]);
        }
        t.w.extend(pack_mma(&mix, 0, C, 2 * C));
        t.w.extend(pack_mma(&flat.w, blk.out.w, C, C));
        t.g.extend(pack_mma_a(&flat.w, blk.pool.w, 2 * C, C));
        for j in 0..C {
            t.b.push(flat.b[blk.mix.b + j] + flat.b[blk.pool.b + j]);
        }
        t.b.extend_from_slice(&flat.b[blk.out.b..blk.out.b + C]);
        for k in 0..2 {
            let n = l.norms[ln_block(i, k)];
            t.ln.extend_from_slice(&flat.ln[n.g..n.g + C]);
            t.ln.extend_from_slice(&flat.ln[n.b..n.b + C]);
        }
    }
    let n = l.norms[LN_TRUNK];
    t.ln.extend_from_slice(&flat.ln[n.g..n.g + C]);
    t.ln.extend_from_slice(&flat.ln[n.b..n.b + C]);
    t
}

const TILE: usize = 1024;
const BOARD_TILE: usize = 1024;
const PAIR_TILE: usize = 32768;

struct Shard {
    call: usize,
    at: usize,
    len: usize,
}

fn shards(sizes: &[usize], from: usize, want: usize) -> Vec<Shard> {
    let mut out = Vec::new();
    let (mut skip, mut taken) = (from, 0);
    for (call, &size) in sizes.iter().enumerate() {
        if taken == want {
            break;
        }
        if skip >= size {
            skip -= size;
            continue;
        }
        let len = (size - skip).min(want - taken);
        out.push(Shard { call, at: skip, len });
        taken += len;
        skip = 0;
    }
    out
}

fn copy<T: Copy>(src: &[T]) -> impl FnOnce(&mut [T]) -> usize + '_ {
    move |dst: &mut [T]| {
        dst[..src.len()].copy_from_slice(src);
        src.len()
    }
}

struct Host<T> {
    buf: Option<cudarc::driver::PinnedHostSlice<T>>,
    len: usize,
    sent: Option<cudarc::driver::CudaEvent>,
}

impl<T> Default for Host<T> {
    fn default() -> Self {
        Host { buf: None, len: 0, sent: None }
    }
}

impl<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Copy> Host<T> {
    fn fill(
        &mut self,
        stream: &Arc<CudaStream>,
        want: usize,
        f: impl FnOnce(&mut [T]) -> usize,
    ) -> Res<()> {
        if let Some(e) = &self.sent {
            e.synchronize().map_err(err)?;
        }
        if self.buf.as_ref().is_none_or(|b| b.len() < want) {
            self.buf = Some(unsafe { stream.context().alloc_pinned::<T>(want.max(1)) }.map_err(err)?);
        }
        let b = self.buf.as_mut().expect("just fitted");
        self.len = f(b.as_mut_slice().map_err(err)?);
        assert!(self.len <= b.len(), "a fill wrote past the buffer");
        Ok(())
    }

    fn send(&mut self, stream: &Arc<CudaStream>, dst: &mut CudaSlice<T>) -> Res<()> {
        if self.len == 0 {
            return Ok(());
        }
        let src = &self.buf.as_ref().expect("a length implies a buffer")
            .as_slice().map_err(err)?[..self.len];
        let mut view = dst.slice_mut(0..self.len);
        stream.memcpy_htod(src, &mut view).map_err(err)?;
        if self.sent.is_none() {
            self.sent = Some(stream.context().new_event(None).map_err(err)?);
        }
        self.sent.as_ref().expect("just made").record(stream).map_err(err)
    }

    fn recv<Src: DevicePtr<T>>(&mut self, stream: &Arc<CudaStream>, src: &Src) -> Res<Vec<T>> {
        let n = src.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if let Some(e) = &self.sent {
            e.synchronize().map_err(err)?;
        }
        if self.buf.as_ref().is_none_or(|b| b.len() < n) {
            self.buf = Some(unsafe { stream.context().alloc_pinned::<T>(n.max(1)) }.map_err(err)?);
        }
        let buf = self.buf.as_mut().expect("just fitted");
        stream.memcpy_dtoh(src, buf).map_err(err)?;
        if self.sent.is_none() {
            self.sent = Some(stream.context().new_event(None).map_err(err)?);
        }
        self.sent.as_ref().expect("just made").record(stream).map_err(err)?;
        self.sent.as_ref().expect("just made").synchronize().map_err(err)?;
        self.len = n;
        Ok(self.buf.as_ref().expect("just fitted").as_slice().map_err(err)?[..n].to_vec())
    }
}

#[derive(Default)]
struct Wire<T> {
    host: Host<T>,
    dev: Arr<T>,
}

impl<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Default + Copy> Wire<T> {
    fn with_cap(stream: &Arc<CudaStream>, cap: usize) -> Res<Wire<T>> {
        let cap = cap.max(1);
        let mut host = Host::default();
        host.fill(stream, cap, |_| 0)?;
        Ok(Wire { host, dev: Arr::with_cap(stream, cap)? })
    }

    fn put(
        &mut self,
        stream: &Arc<CudaStream>,
        want: usize,
        f: impl FnOnce(&mut [T]) -> usize,
    ) -> Res<()> {
        self.host.fill(stream, want, f)?;
        let n = self.host.len;
        self.dev.room(n.max(1))?;
        let dst = self.dev.buf.as_mut().expect("room");
        self.host.send(stream, dst)
    }

    fn buf(&self) -> &CudaSlice<T> {
        self.dev.buf.as_ref().expect("carved")
    }
}

const VALUE_WARPS: usize = 8;

#[derive(Default)]
struct Pack {
    blob: Vec<u32>,
    dst: Vec<u64>,
    at: Vec<u32>,
    src: Vec<u32>,
    sum: Vec<u32>,
    moved: u32,
}

impl Pack {
    fn words(&mut self, w: &[u32]) -> u32 {
        let base = self.blob.len() as u32;
        self.blob.extend_from_slice(w);
        base
    }

    fn clear(&mut self) {
        self.blob.clear();
        self.dst.clear();
        self.at.clear();
        self.src.clear();
        self.sum.clear();
        self.moved = 0;
    }

    fn piece(&mut self, dst: u64, at: u32, src: u32, len: u32) {
        if len == 0 {
            return;
        }
        self.sum.push(self.moved);
        self.moved += len;
        self.dst.push(dst);
        self.at.push(at);
        self.src.push(src);
    }
}

struct Card {
    stream: Arc<CudaStream>,
    busy: parking_lot::Mutex<()>,
    blas: CudaBlas,
    k: Arc<Kernels>,
    solves: Arc<Vec<parking_lot::Mutex<Solve>>>,
    host: parking_lot::Mutex<Stage>,
    pack: parking_lot::Mutex<Pack>,
    down: parking_lot::Mutex<Host<u32>>,
    down_f: parking_lot::Mutex<Host<f32>>,
    batch: parking_lot::Mutex<Batch>,
    scratch: parking_lot::Mutex<Scratch>,
    w: CudaSlice<f32>,
    wh: CudaSlice<f16>,
    value_w: CudaSlice<u32>,
    trunk_w: CudaSlice<u32>,
    trunk_g: CudaSlice<u32>,
    trunk_b: CudaSlice<f32>,
    trunk_ln: CudaSlice<f32>,
    trunk_hex: CudaSlice<u32>,
    b: CudaSlice<f32>,
    ln: CudaSlice<f32>,
    near: CudaSlice<i32>,
    card_facts: CudaSlice<f32>,
    locations: CudaSlice<u8>,
    layout: NetLayout,
}

impl Device {
    pub fn new(ordinals: &[usize], net: &Net, cfg: Cfg, max_slots: usize) -> Res<Device> {
        if ordinals.is_empty() {
            return Err("no cuda device ordinals given".into());
        }
        if net.is_empty() {
            return Err("cannot start the device backend without weights".into());
        }
        let budget = cfg.budget;
        let mut cards = Vec::with_capacity(ordinals.len() * PIPELINE);
        let mut left = max_slots;
        let mut slot_bytes = 0usize;
        for (g, &o) in ordinals.iter().enumerate() {
            let gpu = Gpu::get(o)?;
            gpu.ctx.bind_to_thread().map_err(err)?;
            let mut pair: Vec<Card> = (0..PIPELINE)
                .map(|_| Card::on(&gpu, net))
                .collect::<Res<_>>()?;
            let s0 = Arc::clone(&pair[0].stream);
            s0.context().bind_to_thread().map_err(err)?;
            drop(Solve::at_budget(&s0, &budget)?);
            s0.synchronize().map_err(err)?;
            let free0 = cudarc::driver::result::mem_get_info().map_err(err)?.0 as u64;
            let probe = Solve::at_budget(&s0, &budget)?;
            s0.synchronize().map_err(err)?;
            let measured = free0.saturating_sub(cudarc::driver::result::mem_get_info().map_err(err)?.0 as u64);
            let accounted = probe.bytes() as u64;
            let slot = measured.clamp(accounted, accounted.saturating_mul(2));
            drop(probe);
            s0.synchronize().map_err(err)?;
            if g == 0 {
                slot_bytes = slot as usize;
            }
            let free = cudarc::driver::result::mem_get_info().map_err(err)?.0 as u64;
            let tile = Card::carve_bytes(0, 0, &cfg);
            let extra = Card::carve_bytes(1, 1, &cfg).saturating_sub(tile);
            let per = slot + extra;
            let usable = free.saturating_sub(PIPELINE as u64 * tile);
            let fit = (usable - usable / 10) / per.max(1);
            let gpus_left = ordinals.len() - g;
            const SLOTS_KNEE: usize = 64;
            let mut n = (fit as usize).min(left.div_ceil(gpus_left)).min(SLOTS_KNEE);
            while n > 0 {
                let batch = n.div_ceil(PIPELINE);
                let need = n as u64 * slot + PIPELINE as u64 * Card::carve_bytes(batch, n, &cfg);
                if need + need / 10 <= free {
                    break;
                }
                n -= 1;
            }
            if g == 0 && n == 0 {
                return Err(format!(
                    "a slot of {slot} bytes does not fit in {free} bytes free"
                ));
            }
            let mut solves = Vec::with_capacity(n);
            for _ in 0..n {
                solves.push(Solve::at_budget(&s0, &budget)?);
            }
            let solves: Arc<Vec<parking_lot::Mutex<Solve>>> =
                Arc::new(solves.into_iter().map(parking_lot::Mutex::new).collect());
            let batch = n.div_ceil(PIPELINE);
            for (p, card) in pair.iter_mut().enumerate() {
                card.solves = Arc::clone(&solves);
                card.carve(batch, n, &cfg).map_err(|e| {
                    format!("carve gpu {g} pipe {p} n={n} batch={batch} slot={slot} free={free}: {e}")
                })?;
                card.stream.synchronize().map_err(err)?;
            }
            eprintln!(
                "cuda: gpu {g} carved {n} solve slots (gpu fits {fit}, host allows {}, {} MiB free)",
                left.div_ceil(gpus_left), free >> 20
            );
            left = left.saturating_sub(n);
            cards.extend(pair);
        }
        Ok(Device { cards, n_gpus: ordinals.len(), slot_bytes })
    }

    pub fn cards(&self) -> usize {
        self.n_gpus
    }

    pub fn slots(&self, gpu: usize) -> usize {
        self.cards[gpu * PIPELINE].solves.len()
    }

    pub fn total_slots(&self) -> usize {
        (0..self.n_gpus).map(|g| self.slots(g)).sum()
    }

    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    pub fn slots_per_card(&self) -> usize {
        self.total_slots().checked_div(self.n_gpus).unwrap_or(0)
    }

    pub fn expand_rows(&self, rows: &[u8]) -> Res<Vec<f32>> {
        if rows.len() % ROW_BYTES != 0 {
            return Err("packed rows are not a multiple of ROW_BYTES".into());
        }
        let n = rows.len() / ROW_BYTES;
        if n == 0 {
            return Ok(Vec::new());
        }
        let card = &self.cards[0];
        let _busy = card.busy.lock();
        card.stream.context().bind_to_thread().map_err(err)?;
        let packed = card.stream.memcpy_stod(rows).map_err(err)?;
        let mut out = card.stream.alloc_zeros::<f32>(n * PUBFEAT).map_err(err)?;
        let n_i = n as i32;
        unsafe {
            card.stream
                .launch_builder(&card.k.expand_rows)
                .arg(&packed)
                .arg(&card.card_facts)
                .arg(&card.locations)
                .arg(&mut out)
                .arg(&n_i)
                .launch_unit(LaunchConfig {
                    grid_dim: (n as u32, 1, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(err)?;
        card.stream.memcpy_dtov(&out).map_err(err)
    }

    pub(crate) fn set_weights(&mut self, net: &Net) -> Res<()> {
        if net.is_empty() {
            return Err("cannot publish empty weights to the device".into());
        }
        let flat = net.inference_flat();
        for card in &mut self.cards {
            let _busy = card.busy.lock();
            card.stream.context().bind_to_thread().map_err(err)?;
            card.stream.memcpy_htod(&flat.w, &mut card.w).map_err(err)?;
            card.stream.memcpy_htod(&halved(&flat.w), &mut card.wh).map_err(err)?;
            let t = trunk_pack(&card.layout, &flat);
            card.stream.memcpy_htod(&t.w, &mut card.trunk_w).map_err(err)?;
            card.stream.memcpy_htod(&t.g, &mut card.trunk_g).map_err(err)?;
            card.stream.memcpy_htod(&t.b, &mut card.trunk_b).map_err(err)?;
            card.stream.memcpy_htod(&t.ln, &mut card.trunk_ln).map_err(err)?;
            card.stream.memcpy_htod(&hex_pack(&card.layout, &flat), &mut card.trunk_hex).map_err(err)?;
            let value_w = pack_mma(&flat.w, card.layout.value_out.w, ATTN, BINS);
            card.stream.memcpy_htod(&value_w, &mut card.value_w).map_err(err)?;
            card.stream.memcpy_htod(&flat.b, &mut card.b).map_err(err)?;
            card.stream
                .memcpy_htod(&flat.ln, &mut card.ln)
                .map_err(err)?;
        }
        Ok(())
    }

    pub fn run(&self, calls: &[Call], lane: usize) -> Option<Vec<Reply>> {
        let answered = self.cards[lane].round(calls).inspect_err(|e| {
            eprintln!("cuda: lane {lane}: {e}");
        });
        let mut out: Vec<Reply> = (0..calls.len()).map(|_| Reply::default()).collect();
        for (i, reply) in answered.ok()? {
            out[i] = reply;
        }
        Some(out)
    }

    pub fn resident(&self, card: usize, solve: usize) -> Res<Resident> {
        let c = &self.cards[card];
        let _busy = c.busy.lock();
        c.stream.context().bind_to_thread().map_err(err)?;
        let s = c
            .solves
            .get(solve)
            .ok_or_else(|| format!("solve {solve} is not resident"))?
            .lock();
        let mut h = c.down_f.lock();
        Ok(Resident {
            p: s.ent[Ent::Board as usize].get_f32(
                &c.stream,
                B_P,
                0,
                s.ent[Ent::Board as usize].len() * D,
                &mut h,
            )?,
            query: s.ent[Ent::Config as usize].get_f32(
                &c.stream,
                G_QUERY,
                0,
                s.ent[Ent::Config as usize].len() * ATTN,
                &mut h,
            )?,
            conditioned: halves(&c.stream, &s.conditioned)?,
            value_cfg: halves(&c.stream, &s.value_cfg)?,
            prior: s.ent[Ent::Cell as usize].get_f32(
                &c.stream,
                C_PRIOR,
                0,
                s.ent[Ent::Cell as usize].len(),
                &mut h,
            )?,
            cur: s.ent[Ent::Cell as usize].get_f32(&c.stream, C_CUR, 0, s.ncells, &mut h)?,
            sum: s.ent[Ent::Cell as usize].get_f32(&c.stream, C_SUM, 0, s.ncells, &mut h)?,
            qval: s.ent[Ent::Cell as usize].get_f32(&c.stream, C_QVAL, 0, s.ncells, &mut h)?,
            visits: s.ent[Ent::Cell as usize].get_f32(&c.stream, C_VISITS, 0, s.ncells, &mut h)?,
            reach: s.ent[Ent::Reach as usize].get_f32(&c.stream, R_REACH, 0, s.nreach, &mut h)?,
        })
    }
}

fn halves(stream: &Arc<CudaStream>, a: &Arr<f16>) -> Res<Vec<f32>> {
    let buf = a.buf.as_ref().expect("a capacity implies a buffer");
    let host = stream.memcpy_dtov(&buf.slice(0..a.len)).map_err(err)?;
    Ok(host.into_iter().map(f32::from).collect())
}

pub struct Resident {
    pub p: Vec<f32>,
    pub query: Vec<f32>,
    pub conditioned: Vec<f32>,
    pub value_cfg: Vec<f32>,
    pub prior: Vec<f32>,
    pub cur: Vec<f32>,
    pub sum: Vec<f32>,
    pub qval: Vec<f32>,
    pub visits: Vec<f32>,
    pub reach: Vec<f32>,
}

struct Gpu {
    ctx: Arc<CudaContext>,
    k: Arc<Kernels>,
    torch: Arc<CudaStream>,
}

static GPUS: LazyLock<parking_lot::Mutex<HashMap<usize, Arc<Gpu>>>> =
    LazyLock::new(|| parking_lot::Mutex::new(HashMap::new()));

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
        ATTN,
        BINS,
        HEADS,
        VALUE_WARPS,
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

pub fn expand_rows_torch(
    rows: u64,
    cards: u64,
    locations: u64,
    out: u64,
    n: usize,
    stream: usize,
    ordinal: i32,
) -> Res<()> {
    use cudarc::driver::{result, sys};
    if n == 0 {
        return Ok(());
    }
    let torch_ctx = result::ctx::get_current()
        .map_err(err)?
        .ok_or("PyTorch has no current CUDA context")?;
    let device = result::device::get(ordinal).map_err(err)?;
    let primary = unsafe { result::primary_ctx::retain(device) }.map_err(err)?;
    let same_context = primary == torch_ctx;
    unsafe { result::primary_ctx::release(device) }.map_err(err)?;
    if !same_context {
        return Err("PyTorch does not use the device primary CUDA context".into());
    }
    let gpu = Gpu::get(ordinal as usize)?;

    let before = gpu.ctx.new_event(None).map_err(err)?;
    unsafe {
        result::event::record(before.cu_event(), stream as sys::CUstream).map_err(err)?;
    }
    gpu.torch.wait(&before).map_err(err)?;

    let rows = unsafe { gpu.torch.upgrade_device_ptr::<u8>(rows, n * ROW_BYTES) };
    let cards = unsafe { gpu.torch.upgrade_device_ptr::<f32>(cards, N_UNITS * CARD_FEATS) };
    let locations = unsafe { gpu.torch.upgrade_device_ptr::<u8>(locations, N_HEXES) };
    let mut out = unsafe { gpu.torch.upgrade_device_ptr::<f32>(out, n * PUBFEAT) };
    let n_i = n as i32;
    unsafe {
        gpu.torch
            .launch_builder(&gpu.k.expand_rows)
            .arg(&rows)
            .arg(&cards)
            .arg(&locations)
            .arg(&mut out)
            .arg(&n_i)
            .launch_unit(LaunchConfig {
                grid_dim: (n as u32, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
            .map_err(err)?;
    }
    rows.leak();
    cards.leak();
    locations.leak();
    out.leak();

    let after = gpu.torch.record_event(None).map_err(err)?;
    unsafe {
        result::stream::wait_event(
            stream as sys::CUstream,
            after.cu_event(),
            sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
        )
        .map_err(err)
    }
}

impl Gpu {
    fn get(ordinal: usize) -> Res<Arc<Gpu>> {
        let mut gpus = GPUS.lock();
        if let Some(gpu) = gpus.get(&ordinal) {
            gpu.ctx.bind_to_thread().map_err(err)?;
            return Ok(Arc::clone(gpu));
        }
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("device {ordinal}: {e:?}"))?;
        unsafe { ctx.disable_event_tracking() };
        let (major, minor) = ctx.compute_capability().map_err(err)?;
        let source = slot::tree_source() + KERNELS;
        let opts = compile_options(major, minor);
        let ptx = compile_ptx_with_opts(&source, opts)
            .map_err(|e| format!("nvrtc: {e:?}"))?;
        let module = ctx.load_module(ptx).map_err(err)?;
        let k = Kernels::load(&module)?;
        k.trunk
            .set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                TRUNK_SHARED as i32,
            )
            .map_err(err)?;
        {
            use cudarc::driver::sys::CUfunction_attribute_enum as A;
            let regs = k.trunk.get_attribute(A::CU_FUNC_ATTRIBUTE_NUM_REGS).map_err(err)?;
            let local = k.trunk.get_attribute(A::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES).map_err(err)?;
            eprintln!("k_trunk: regs={regs} local_bytes={local}");
        }
        let torch = ctx.new_stream().map_err(err)?;
        let gpu = Arc::new(Gpu { ctx, k: Arc::new(k), torch });
        gpus.insert(ordinal, Arc::clone(&gpu));
        Ok(gpu)
    }
}

impl Card {
    fn carve_bytes(n: usize, slots: usize, cfg: &Cfg) -> u64 {
        Plan { stream: None, bytes: 0 }.build(n, slots, cfg).map_or(0, |(_, _, _, b)| b as u64)
    }

    fn on(gpu: &Gpu, net: &Net) -> Res<Card> {
        let stream = gpu.ctx.new_stream().map_err(err)?;
        let blas = CudaBlas::new(stream.clone()).map_err(err)?;
        let flat = net.inference_flat();
        let near: Vec<i32> = board()
            .neighbors
            .iter()
            .flatten()
            .map(|&n| if n == NONE { -1 } else { n as i32 })
            .collect();
        let mut card_facts = vec![0.0; N_UNITS * CARD_FEATS];
        for u in 0..N_UNITS {
            write_card_features(
                u as u8,
                &mut card_facts[u * CARD_FEATS..(u + 1) * CARD_FEATS],
            );
        }
        let locations: Vec<u8> = board().is_location.iter().map(|&x| x as u8).collect();
        let layout = NetLayout::new();
        let t = trunk_pack(&layout, &flat);
        Ok(Card {
            value_w: stream.memcpy_stod(&pack_mma(&flat.w, layout.value_out.w, ATTN, BINS))
                .map_err(err)?,
            wh: stream.memcpy_stod(&halved(&flat.w)).map_err(err)?,
            trunk_w: stream.memcpy_stod(&t.w).map_err(err)?,
            trunk_g: stream.memcpy_stod(&t.g).map_err(err)?,
            trunk_b: stream.memcpy_stod(&t.b).map_err(err)?,
            trunk_ln: stream.memcpy_stod(&t.ln).map_err(err)?,
            trunk_hex: stream.memcpy_stod(&hex_pack(&layout, &flat)).map_err(err)?,
            w: stream.memcpy_stod(&flat.w).map_err(err)?,
            b: stream.memcpy_stod(&flat.b).map_err(err)?,
            ln: stream.memcpy_stod(&flat.ln).map_err(err)?,
            near: stream.memcpy_stod(&near).map_err(err)?,
            card_facts: stream.memcpy_stod(&card_facts).map_err(err)?,
            locations: stream.memcpy_stod(&locations).map_err(err)?,
            stream,
            busy: parking_lot::Mutex::new(()),
            blas,
            k: Arc::clone(&gpu.k),
            solves: Arc::new(Vec::new()),
            host: parking_lot::Mutex::new(Stage::default()),
            pack: parking_lot::Mutex::new(Pack::default()),
            down: parking_lot::Mutex::new(Host::default()),
            down_f: parking_lot::Mutex::new(Host::default()),
            batch: parking_lot::Mutex::new(Batch::default()),
            scratch: parking_lot::Mutex::new(Scratch::default()),
            layout,
        })
    }

    fn carve(&mut self, n: usize, slots: usize, cfg: &Cfg) -> Res<()> {
        if n == 0 {
            return Ok(());
        }
        self.stream.context().bind_to_thread().map_err(err)?;
        let plan = Plan { stream: Some(Arc::clone(&self.stream)), bytes: 0 };
        let (scratch, stage, batch, _) = plan.build(n, slots, cfg)?;
        *self.host.lock() = stage;
        *self.scratch.lock() = scratch;
        *self.batch.lock() = batch;
        Ok(())
    }

    fn round(&self, calls: &[Call]) -> Res<Vec<(usize, Reply)>> {
        let mine: Vec<usize> = (0..calls.len()).collect();
        let mine = &mine[..];
        let _busy = self.busy.lock();
        self.stream.context().bind_to_thread().map_err(err)?;
        let mut slots: Vec<usize> = mine.iter().map(|&i| calls[i].solve()).collect();
        slots.sort_unstable();
        slots.dedup();
        {
            for &slot in &slots {
                let solve = self.solves.get(slot).ok_or_else(|| {
                    format!("round names solve slot {slot}, but only {} were carved",
                            self.solves.len())
                })?;
                self.stream.wait(&solve.lock().ready).map_err(err)?;
            }
        }
        let pick = |kind: usize| -> Vec<usize> {
            mine.iter()
                .copied()
                .filter(|&i| calls[i].kind() == kind)
                .collect()
        };
        let trees = pick(2);
        {
            for &i in &trees {
                let Call::Tree { solve, fresh, .. } = &calls[i] else { unreachable!() };
                if *fresh {
                    self.slot(*solve).rewind(&self.stream)?;
                }
            }
        }
        let mut out = Vec::with_capacity(mine.len());
        fn at(stage: &'static str) -> impl Fn(String) -> String {
            move |e| format!("{stage}: {e}")
        }
        let mut pack = self.pack.lock();
        pack.clear();
        self.trunk(calls, &pick(0), &mut pack).map_err(at("trunk"))?;
        self.configs(calls, &pick(1)).map_err(at("configs"))?;
        self.tree(calls, &trees, &mut pack).map_err(at("tree"))?;
        self.scatter(&mut pack).map_err(at("scatter"))?;
        drop(pack);
        self.priors(calls, &trees).map_err(at("priors"))?;
        let rule = |i| match &calls[i] {
            Call::Iterate { cfr, puct, .. } => [
                cfr.alpha.to_bits(), cfr.beta.to_bits(), cfr.gamma.to_bits(),
                cfr.predict.to_bits(), puct.to_bits(),
            ],
            _ => unreachable!(),
        };
        let mut iterates = pick(3);
        iterates.sort_unstable_by_key(|&i| rule(i));
        for same in iterates.chunk_by(|a, b| rule(*a) == rule(*b)) {
            self.iterate(calls, same, &mut out).map_err(at("iterate"))?;
        }
        self.read(calls, &pick(4), &mut out).map_err(at("read"))?;
        {
            for &slot in &slots {
                self.solves[slot].lock().ready.record(&self.stream).map_err(err)?;
            }
        }
        Ok(out)
    }

    fn gemm<I: Elem, O: Elem>(
        &self,
        s: Span,
        inp: &impl DevicePtr<I>,
        rows: usize,
        beta: f32,
        out: &mut impl DevicePtrMut<O>,
        compute: cudarc::cublas::sys::cublasComputeType_t,
    ) -> Res<()> {
        if rows == 0 {
            return Ok(());
        }
        let (m, n, k) = (s.o as i32, rows as i32, s.i as i32);
        let one = 1.0f32;
        let (a, wt) = if I::HALF {
            (self.wh.slice(s.w..s.w + s.i * s.o).device_ptr(&self.stream).0, f16::KIND)
        } else {
            (self.w.slice(s.w..s.w + s.i * s.o).device_ptr(&self.stream).0, f32::KIND)
        };
        let (b, _kb) = inp.device_ptr(&self.stream);
        let (c, _kc) = out.device_ptr_mut(&self.stream);
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *self.blas.handle(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                (&one) as *const f32 as *const _, a as *const _, wt, m,
                b as *const _, I::KIND, k, (&beta) as *const f32 as *const _,
                c as *mut _, O::KIND, m, compute,
                cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        }
        .map_err(err)
    }

    fn lin(&self, s: Span, inp: &impl DevicePtr<f32>, rows: usize, beta: f32, out: &mut impl DevicePtrMut<f32>) -> Res<()> {
        self.gemm(s, inp, rows, beta, out, cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F)
    }

    fn hlin(&self, s: Span, inp: &impl DevicePtr<f16>, rows: usize, beta: f32, out: &mut impl DevicePtrMut<f16>) -> Res<()> {
        self.gemm(s, inp, rows, beta, out, cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F)
    }

    fn hlin32(&self, s: Span, inp: &impl DevicePtr<f16>, rows: usize, out: &mut impl DevicePtrMut<f32>) -> Res<()> {
        self.gemm(s, inp, rows, 0.0, out, cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F)
    }

    fn bias(&self, s: Span, rows: usize, out: &mut CudaSlice<f32>) -> Res<()> {
        if s.b == usize::MAX || rows == 0 {
            return Ok(());
        }
        let bias = self.b.slice(s.b..s.b + s.o);
        let (rows_i, width) = (rows as i32, s.o as i32);
        unsafe {
            self.stream
                .launch_builder(&self.k.bias)
                .arg(out).arg(&bias).arg(&rows_i).arg(&width)
                .launch_unit(warp_rows(rows))
        }
        .map_err(err)
    }

    fn run<A: DevicePtr<f32>>(
        &self,
        s: Span,
        inp: &A,
        rows: usize,
        out: &mut CudaSlice<f32>,
    ) -> Res<()> {
        self.lin(s, inp, rows, 0.0, out)?;
        self.bias(s, rows, out)
    }

    fn norm(&self, s: NormSpan, rows: usize, act: bool, x: &mut CudaSlice<f32>) -> Res<()> {
        if rows == 0 {
            return Ok(());
        }
        let g = self.ln.slice(s.g..s.g + s.width);
        let b = self.ln.slice(s.b..s.b + s.width);
        let (rows_i, width, act) = (rows as i32, s.width as i32, act as i32);
        unsafe {
            self.stream
                .launch_builder(&self.k.norm_ip)
                .arg(x).arg(&g).arg(&b)
                .arg(&rows_i).arg(&width).arg(&act)
                .launch_unit(warp_rows(rows))
        }
        .map_err(err)
    }

    fn slot(&self, solve: usize) -> parking_lot::MutexGuard<'_, Solve> {
        let n = self.solves.len();
        self.solves
            .get(solve)
            .unwrap_or_else(|| panic!("solve {solve} pinned to a card that holds {n} slots"))
            .lock()
    }

    fn trunk(&self, calls: &[Call], mine: &[usize], pack: &mut Pack) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        let each = |i: usize| -> (&[u8], &[f32], usize) {
            let Call::Trunk { packed, cards, boards, .. } = &calls[i] else {
                unreachable!("trunk shard holds only trunk calls")
            };
            assert_eq!(packed.len(), boards * ROW_BYTES, "trunk input is not one packed row a board");
            assert_eq!(cards.len(), NTYPE * TYPE, "trunk card table");
            (packed, cards, *boards)
        };
        let sizes: Vec<usize> = mine.iter().map(|&i| each(i).2).collect();
        let rows: usize = sizes.iter().sum();
        let s = &self.stream;
        {
            let mut stage = self.host.lock();
            stage.cards.put(s, mine.len() * NTYPE * TYPE, |dst| {
                let mut at = 0;
                for &i in mine {
                    let (_, cd, _) = each(i);
                    dst[at..at + cd.len()].copy_from_slice(cd);
                    at += cd.len();
                }
                at
            })?;
        }
        let mut board0 = 0usize;
        while board0 < rows {
            let n = TILE.min(rows - board0);
            {
                let mut stage = self.host.lock();
                let tile = shards(&sizes, board0, n);
                stage.packed.put(s, n * ROW_BYTES, |dst| {
                    let mut wrote = 0;
                    for w in &tile {
                        let packed = each(mine[w.call]).0;
                        let a = w.at * ROW_BYTES;
                        dst[wrote..wrote + w.len * ROW_BYTES]
                            .copy_from_slice(&packed[a..a + w.len * ROW_BYTES]);
                        wrote += w.len * ROW_BYTES;
                    }
                    wrote
                })?;
                stage.card_of_row.put(s, n, |dst| {
                    let mut wrote = 0;
                    for w in &tile {
                        dst[wrote..wrote + w.len].fill(w.call as i32);
                        wrote += w.len;
                    }
                    wrote
                })?;
                let rows_i = n as i32;
                let Stage { packed, xpub, .. } = &mut *stage;
                let packed = packed.dev.buf.as_ref().expect("staged");
                let xpub = xpub.buf.as_mut().expect("carved");
                unsafe {
                    self.stream
                        .launch_builder(&self.k.expand_rows)
                        .arg(packed)
                        .arg(&self.card_facts)
                        .arg(&self.locations)
                        .arg(xpub)
                        .arg(&rows_i)
                        .launch_unit(LaunchConfig {
                            grid_dim: (n as u32, 1, 1),
                            block_dim: (THREADS, 1, 1),
                            shared_mem_bytes: 0,
                        })
                }
                .map_err(err)?;
            }
            self.trunk_tile(calls, mine, &sizes, board0, n)?;
            board0 += n;
        }
        self.keep(calls, mine, pack)
    }

    fn encode_boards(&self, n: usize) -> Res<()> {
        let stage = self.host.lock();
        let xpub = stage.xpub.buf.as_ref().expect("carved");
        let cards = stage.cards.dev.buf.as_ref().expect("staged");
        let card_of_row = stage.card_of_row.dev.buf.as_ref().expect("staged");
        let l = &self.layout;
        let mut sc = self.scratch.lock();
        sc.spatial.room(n * N_HEXES * C)?;
        sc.token_half.room(n * NTYPE * C)?;
        sc.input.room(n * (2 * C + LOOSE))?;
        let Scratch { token_half, input, spatial, .. } = &mut *sc;
        carved!(token_half, input, spatial);
        let w = |at: usize, len: usize| self.w.slice(at..at + len).device_ptr(&self.stream).0;
        let stem = Stem {
            cards: cards.device_ptr(&self.stream).0,
            card_of_row: card_of_row.device_ptr(&self.stream).0,
            w_pile: w(l.pile.w, PILE_COUNTS * TYPE),
            seat: w(l.seat, 2 * TYPE),
            w_tok: w(l.tok_stem.w, TYPE * C),
            w_hex: self.trunk_hex.device_ptr(&self.stream).0,
            b_hex: self.b.slice(l.hex_stem.b..l.hex_stem.b + C).device_ptr(&self.stream).0,
            pos: w(l.pos, N_HEXES * C),
            w_glob: w(l.glob_stem.w, LOOSE * C),
        };
        let rows_i = n as i32;
        unsafe {
            self.stream
                .launch_builder(&self.k.trunk)
                .arg(xpub).arg(&stem).arg(&self.trunk_w).arg(&self.trunk_g).arg(&self.trunk_b)
                .arg(&self.trunk_ln).arg(&self.near).arg(&mut *input)
                .arg(&mut *spatial).arg(&mut *token_half).arg(&rows_i)
                .launch_unit(LaunchConfig {
                    grid_dim: (n.div_ceil(TRUNK_BOARDS) as u32, 1, 1),
                    block_dim: (TRUNK_THREADS, 1, 1),
                    shared_mem_bytes: TRUNK_SHARED,
                })
        }
        .map_err(err)
    }

    fn trunk_tile(
        &self,
        calls: &[Call],
        mine: &[usize],
        sizes: &[usize],
        board0: usize,
        n: usize,
    ) -> Res<()> {
        self.encode_boards(n)?;
        let s = &self.stream;
        let mut sc = self.scratch.lock();
        sc.h.room(n * D)?;
        sc.z.room(n * ATTN)?;
        let Scratch { input, token_half, spatial, h, z, .. } = &mut *sc;
        let p = h.buf.as_mut().unwrap();
        self.run(self.layout.board_out, input.buf.as_ref().unwrap(), n, p)?;
        let vh = self.layout.value_hidden;
        let value_board = Span { w: vh.w + ATTN * ATTN, b: vh.b, i: D, o: ATTN };
        self.run(value_board, p, n, z.buf.as_mut().unwrap())?;
        let mut src = 0;
        for w in shards(sizes, board0, n) {
            let Call::Trunk {
                solve, boards_at, ..
            } = &calls[mine[w.call]]
            else {
                unreachable!("trunk shard holds only trunk calls")
            };
            self.slot(*solve).copy_board(
                s,
                *boards_at + w.at,
                p,
                z.buf.as_ref().unwrap(),
                token_half.buf.as_ref().unwrap(),
                spatial.buf.as_ref().unwrap(),
                src,
                w.len,
            )?;
            src += w.len;
        }
        Ok(())
    }

    fn keep(&self, calls: &[Call], mine: &[usize], pack: &mut Pack) -> Res<()> {
        for &i in mine {
            let Call::Trunk { solve, at: row0, queries: nrows, board_of, coff, .. } = &calls[i]
            else {
                unreachable!("trunk shard holds only trunk calls")
            };
            let nrows = *nrows;
            let mut b = self.slot(*solve);
            b.reserve(Ent::Row, row0 + nrows)?;
            let words = pack.words(board_of);
            let dst = b.ent[Ent::Row as usize].field(Y_BOARD_OF, &self.stream);
            pack.piece(dst, *row0 as u32, words, nrows as u32);
            let base = b.cells as u32;
            let shifted: Vec<u32> = coff.iter().map(|x| x + base).collect();
            b.host_coff.extend(shifted.iter().skip(1));
            let grown = *coff.last().expect("a coff run ends with its length") as usize;
            let cells = b.cells + grown;
            b.reserve(Ent::Cidx, cells)?;
            let words = pack.words(&shifted);
            let dst = b.ent[Ent::Row as usize].field(Y_COFF, &self.stream);
            pack.piece(dst, 2 * *row0 as u32, words, shifted.len() as u32);
            b.cells += grown;
            b.rows = row0 + nrows;
        }
        Ok(())
    }

    fn configs(&self, calls: &[Call], mine: &[usize]) -> Res<()> {
        if mine.is_empty() { return Ok(()); }
        let sizes: Vec<usize> = mine.iter().map(|&i| match &calls[i] {
            Call::Configs { n, .. } => *n,
            _ => unreachable!("config shard holds only config calls"),
        }).collect();
        let total: usize = sizes.iter().sum();
        let mut phi = Vec::with_capacity(total * CFEAT);
        let mut owner = Vec::with_capacity(total);
        let mut cards = Vec::new();
        let mut solves = Vec::with_capacity(mine.len());
        for (part, &i) in mine.iter().enumerate() {
            let Call::Configs { solve, phi: p, owner: o, cards: c, n, .. } = &calls[i] else { unreachable!() };
            assert_eq!(p.len(), n * CFEAT);
            assert_eq!(o.len(), *n);
            assert_eq!(c.len(), NTYPE * TYPE);
            assert!(o.iter().all(|&x| x < 2));
            phi.extend_from_slice(p);
            owner.extend(o.iter().map(|&x| x + 2 * part as u32));
            cards.extend_from_slice(c);
            solves.push(*solve);
        }
        let stream = &self.stream;
        let mut stage = self.host.lock();
        stage.phi.put(stream, phi.len(), copy(&phi))?;
        stage.owner.put(stream, owner.len(), copy(&owner))?;
        stage.cfg_cards.put(stream, cards.len(), copy(&cards))?;
        drop(stage);
        for at in (0..total).step_by(TILE) {
            let n = TILE.min(total - at);
            self.config_tile(calls, mine, &sizes, at, n)?;
        }
        let mut pairs = Vec::new();
        for (part, &i) in mine.iter().enumerate() {
            let Call::Configs { solve, pair_at, pair_board, pair_config, .. } = &calls[i] else { unreachable!() };
            assert_eq!(pair_board.len(), pair_config.len());
            let mut slot = self.slot(*solve);
            slot.conditioned.room((pair_at + pair_board.len()) * ATTN)?;
            slot.value_cfg.room((pair_at + pair_board.len()) * ATTN)?;
            pairs.extend((0..pair_board.len()).map(|j| {
                (part as u32, pair_at + j, pair_board[j], pair_config[j])
            }));
        }
        pairs.sort_unstable_by_key(|p| (p.0, p.2));
        let mut boards = Vec::new();
        let mut offsets = Vec::new();
        for (i, pair) in pairs.iter_mut().enumerate() {
            let board = (pair.0, pair.2);
            if boards.last() != Some(&board) {
                offsets.push(i);
                boards.push(board);
            }
            pair.2 = (boards.len() - 1) as u32;
        }
        offsets.push(pairs.len());
        self.lay(&solves)?;
        for at in (0..boards.len()).step_by(BOARD_TILE) {
            let end = (at + BOARD_TILE).min(boards.len());
            self.attention_boards(at as u32, &boards[at..end], &pairs[offsets[at]..offsets[end]])?;
        }
        Ok(())
    }

    fn config_tile(&self, calls: &[Call], mine: &[usize], sizes: &[usize], cfg0: usize, n: usize) -> Res<()> {
        let stream = &self.stream;
        let stage = self.host.lock();
        let phi = stage.phi.dev.buf.as_ref().expect("staged");
        let owner = stage.owner.dev.buf.as_ref().expect("staged");
        let cards = stage.cfg_cards.dev.buf.as_ref().expect("staged");
        let width = 3 + TYPE;
        let mut scratch = self.scratch.lock();
        scratch.tokens.room(n * NSLOT * width)?;
        scratch.projected.room(n * NSLOT * ATTN)?;
        scratch.facts.room(n * ATTN)?;
        scratch.h.room(n * ATTN)?;
        let Scratch { tokens, projected, facts, h, .. } = &mut *scratch;
        carved!(tokens, projected, facts, h);
        let phi = phi.slice(cfg0 * CFEAT..);
        let owner = owner.slice(cfg0..);
        let (n_i, nslot, cfeat) = (n as i32, NSLOT as i32, CFEAT as i32);
        let (ntype, type_i, attn) = (NTYPE as i32, TYPE as i32, ATTN as i32);
        launch!(self, cfg_slots, n * NSLOT * width, &phi, &owner, cards, &mut *tokens,
                &n_i, &nslot, &cfeat, &ntype, &type_i)?;
        self.run(self.layout.cfg_in, tokens, n * NSLOT, projected)?;
        let hidden = (n * NSLOT * ATTN) as i32;
        launch!(self, gelu, n * NSLOT * ATTN, &mut *projected, &hidden)?;
        launch!(self, sum_slots, n * ATTN, &*projected, &mut *facts,
                &n_i, &nslot, &attn)?;
        let seat = self.w.slice(self.layout.cfg_seat..self.layout.cfg_seat + 2 * ATTN);
        launch!(self, cfg_seat, n * ATTN, &mut *facts, &owner, &seat, &n_i, &attn)?;
        self.norm(self.layout.norms[LN_CFG], n, true, facts)?;
        self.run(self.layout.attn_q, facts, n, h)?;
        let mut src = 0;
        for shard in shards(sizes, cfg0, n) {
            let Call::Configs { solve, at, .. } = &calls[mine[shard.call]] else { unreachable!() };
            self.slot(*solve).copy_cfg(
                stream, *at + shard.at, facts, h, src, shard.len)?;
            src += shard.len;
        }
        Ok(())
    }

    fn attention_boards(&self, board_at: u32, boards: &[(u32, u32)], pairs: &[(u32, usize, u32, u32)]) -> Res<()> {
        {
            let stream = &self.stream;
            let mut stage = self.host.lock();
            stage.board_part.put(stream, boards.len(), |dst| {
                for (out, board) in dst.iter_mut().zip(boards) { *out = board.0; }
                boards.len()
            })?;
            stage.board_local.put(stream, boards.len(), |dst| {
                for (out, board) in dst.iter_mut().zip(boards) { *out = board.1; }
                boards.len()
            })?;
            let batch = self.batch.lock();
            let mut scratch = self.scratch.lock();
            let ntokens = N_HEXES + NTYPE;
            scratch.public.room(boards.len() * ntokens * C)?;
            scratch.kv.room(boards.len() * ntokens * 2 * ATTN)?;
            let Scratch { public, kv, .. } = &mut *scratch;
            carved!(public, kv);
            let (nb, tokens, c) = (boards.len() as i32, ntokens as i32, C as i32);
            unsafe {
                stream.launch_builder(&self.k.public_cache)
                    .arg(batch.trees.buf()).arg(stage.board_part.buf()).arg(stage.board_local.buf())
                    .arg(&mut *public).arg(&nb).arg(&tokens).arg(&c)
                    .launch_unit(spread(boards.len() * ntokens * C))
            }.map_err(err)?;
            self.hlin(self.layout.attention_kv(), &*public, boards.len() * ntokens, 0.0, kv)?;
        }
        for tile in pairs.chunks(PAIR_TILE) {
            self.attention_pairs(board_at, tile)?;
        }
        Ok(())
    }

    fn attention_pairs(&self, board_at: u32, pairs: &[(u32, usize, u32, u32)]) -> Res<()> {
        let stream = &self.stream;
        let mut stage = self.host.lock();
        stage.pair_part.put(stream, pairs.len(), |dst| {
            for (out, pair) in dst.iter_mut().zip(pairs) { *out = pair.0; }
            pairs.len()
        })?;
        stage.pair_at.put(stream, pairs.len(), |dst| {
            for (out, pair) in dst.iter_mut().zip(pairs) { *out = pair.1 as u32; }
            pairs.len()
        })?;
        stage.pair_board.put(stream, pairs.len(), |dst| {
            for (out, pair) in dst.iter_mut().zip(pairs) { *out = pair.2 - board_at; }
            pairs.len()
        })?;
        stage.pair_config.put(stream, pairs.len(), |dst| {
            for (out, pair) in dst.iter_mut().zip(pairs) { *out = pair.3; }
            pairs.len()
        })?;
        stage.pair_start.put(stream, pairs.len() + 1, |dst| {
            let mut n = 0;
            for (i, pair) in pairs.iter().enumerate() {
                if i == 0 || pairs[i - 1].2 != pair.2 {
                    dst[n] = i as u32;
                    n += 1;
                }
            }
            dst[n] = pairs.len() as u32;
            n + 1
        })?;
        let runs = stage.pair_start.host.len - 1;
        let batch = self.batch.lock();
        let mut scratch = self.scratch.lock();
        scratch.attn_mix.room(pairs.len() * ATTN)?;
        scratch.attn_out.room(pairs.len() * ATTN)?;
        scratch.attn_val.room(pairs.len() * ATTN)?;
        let Scratch { kv, attn_mix, attn_out, attn_val, .. } = &mut *scratch;
        carved!(kv, attn_mix, attn_out, attn_val);
        let part = stage.pair_part.buf();
        let config = stage.pair_config.buf();
        let n = pairs.len() as i32;
        let kvb = self.layout.attention_kv().b;
        let kv_b = self.b.slice(kvb..kvb + 2 * ATTN);
        unsafe {
            stream.launch_builder(&self.k.attention)
                .arg(batch.trees.buf()).arg(&*kv).arg(&kv_b)
                .arg(part).arg(stage.pair_board.buf()).arg(config).arg(&mut *attn_mix)
                .arg(stage.pair_start.buf())
                .launch_unit(LaunchConfig {
                    grid_dim: (runs as u32, 1, 1),
                    block_dim: (32, HEADS as u32, 1),
                    shared_mem_bytes: 0,
                })
        }.map_err(err)?;
        self.hlin(self.layout.attn_out, &*attn_mix, pairs.len(), 0.0, attn_out)?;
        let norm = self.layout.norms[LN_ATTN];
        let gamma = self.ln.slice(norm.g..norm.g + ATTN);
        let beta = self.ln.slice(norm.b..norm.b + ATTN);
        unsafe {
            stream.launch_builder(&self.k.attention_residual)
                .arg(batch.trees.buf()).arg(&mut *attn_out).arg(part).arg(config)
                .arg(&gamma).arg(&beta).arg(&n)
                .launch_unit(warp_rows(pairs.len()))
        }.map_err(err)?;
        let value_cfg = Span { w: self.layout.value_hidden.w, b: usize::MAX, i: ATTN, o: ATTN };
        self.hlin(value_cfg, &*attn_out, pairs.len(), 0.0, attn_val)?;
        launch!(self, attention_store, pairs.len() * ATTN,
                batch.trees.buf(), part, stage.pair_at.buf(), &*attn_out, &*attn_val, &n)?;
        Ok(())
    }

    fn priors(&self, calls: &[Call], mine: &[usize]) -> Res<()> {
        let each = |i: usize| -> (&[Prime], &[u32], &[u32], f32) {
            let Call::Tree { prime, acts, cells, prior_temp, .. } = &calls[i] else {
                unreachable!("tree shard holds only tree calls")
            };
            (prime, acts, cells, *prior_temp)
        };
        let solves: Vec<usize> = mine
            .iter()
            .copied()
            .filter(|&i| !each(i).0.is_empty())
            .map(|i| calls[i].solve())
            .collect();
        if solves.is_empty() {
            return Ok(());
        }
        let mut want: Vec<(u32, u32, Prime)> = Vec::new();
        let mut desc: Vec<u32> = Vec::new();
        let mut cells: Vec<u32> = Vec::new();
        for (p, &i) in mine.iter().filter(|&&i| !each(i).0.is_empty()).enumerate() {
            let (prime, a, c, temp) = each(i);
            let inv = (1.0f32 / temp.max(1e-6)).to_bits();
            let at = (desc.len() / crate::search::ACT_BYTES) as u32;
            let cell_at = cells.len() as u32;
            want.extend(prime.iter().map(|q| {
                let q = Prime { at: at + q.at, cell_at: cell_at + q.cell_at, ..*q };
                (p as u32, inv, q)
            }));
            desc.extend_from_slice(a);
            cells.extend_from_slice(c);
        }
        let m = want.len();
        self.lay(&solves)?;
        let mut batch = self.batch.lock();
        let mut i = 0usize;
        while i < m {
            let mut j = i;
            let mut na_c = 0usize;
            let mut wide = 0u32;
            while j < m && (j - i) < TILE && na_c + want[j].2.na as usize <= TILE {
                na_c += want[j].2.na as usize;
                wide = wide.max(want[j].2.nc);
                j += 1;
            }
            if j == i {
                na_c = want[j].2.na as usize;
                wide = want[j].2.nc;
                j = i + 1;
            }
            let (act0, cell0) = (want[i].2.at as usize, want[i].2.cell_at as usize);
            let cell1 = if j < m { want[j].2.cell_at as usize } else { cells.len() };
            let col = |f: &dyn Fn(&(u32, u32, Prime)) -> u32| -> Vec<u32> {
                want[i..j].iter().map(f).collect()
            };
            let act_node: Vec<u32> = (0..j - i)
                .flat_map(|k| std::iter::repeat_n(k as u32, want[i + k].2.na as usize))
                .collect();
            let flat: Vec<u32> = [
                col(&|r| r.0),
                col(&|r| r.2.node),
                col(&|r| r.2.row),
                col(&|r| r.2.at - want[i].2.at),
                col(&|r| r.2.cell_at - want[i].2.cell_at),
                col(&|r| r.1),
                act_node,
                desc[act0 * crate::search::ACT_BYTES..(act0 + na_c) * crate::search::ACT_BYTES].to_vec(),
                cells[cell0..cell1].to_vec(),
            ]
            .concat();
            self.prior_tile(&mut batch, &flat, j - i, na_c, cell1 - cell0, wide)?;
            i = j;
        }
        Ok(())
    }

    fn prior_tile(
        &self,
        batch: &mut Batch,
        flat: &[u32],
        m: usize,
        na: usize,
        ncells: usize,
        widest: u32,
    ) -> Res<()> {
        let s = &self.stream;
        batch.prime.put(s, flat.len(), copy(flat))?;
        let dev = batch.prime.buf();
        let at = |k: usize, n: usize| dev.slice(k..k + n);
        let (part_d, node_d, row_d) = (at(0, m), at(m, m), at(2 * m, m));
        let (act_at_d, cell_at_d, inv_d) = (at(3 * m, m), at(4 * m, m), at(5 * m, m));
        let act_node_d = at(6 * m, na);
        let desc_d = at(6 * m + na, crate::search::ACT_BYTES * na);
        let cells_d = at(6 * m + (1 + crate::search::ACT_BYTES) * na, ncells);
        let l = &self.layout;
        let (na_i, m_i, d_i, c_i, attn_i) = (na as i32, m as i32, D as i32, C as i32, ATTN as i32);
        let (ntype, nhex) = (NTYPE as i32, N_HEXES as i32);
        let mut sc = self.scratch.lock();
        sc.action.room(na * C)?;
        sc.h.room((m * D).max(na * D))?;
        sc.facts.room(m * D)?;
        sc.input.room((m * (D + 2 * ATTN)).max(na * ATTN))?;
        sc.queries.room(m * 2 * ATTN)?;
        sc.z.room(m * C)?;
        sc.pooled.room(m * C)?;
        let Scratch {
            action,
            h,
            input,
            queries,
            z,
            pooled,
            facts,
            ..
        } = &mut *sc;
        carved!(action, h, input, queries, z, pooled, facts);
        let kind = self
            .w
            .slice(l.act_kind..l.act_kind + crate::actions::N_KINDS * C);
        let role = self.w.slice(l.act_role..l.act_role + 5 * C);
        launch!(
            self,
            act_feats,
            na * C,
            batch.trees.buf(),
            &part_d,
            &row_d,
            &desc_d,
            &act_node_d,
            &kind,
            &role,
            &mut *action,
            &na_i,
            &ntype,
            &nhex,
            &c_i
        )?;
        unsafe {
            self.stream
                .launch_builder(&self.k.policy_inputs)
                .arg(batch.trees.buf())
                .arg(&part_d)
                .arg(&node_d)
                .arg(&row_d)
                .arg(&mut *input)
                .arg(&mut *queries)
                .arg(&mut *facts)
                .arg(&m_i)
                .arg(&attn_i)
                .arg(&d_i)
                .launch_unit(LaunchConfig {
                    grid_dim: (m as u32, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .map_err(err)?;
        self.run(l.act_board, facts, m, pooled)?;
        self.run(l.head_policy, input, m, h)?;
        self.run(l.act_h, h, m, z)?;
        let policy_belief = Span {
            w: l.cfg_policy.w + ATTN * D,
            b: usize::MAX,
            i: 2 * ATTN,
            o: D,
        };
        self.lin(policy_belief, queries, m, 0.0, facts)?;
        launch!(
            self,
            act_add,
            na * C,
            &mut *action,
            &*pooled,
            &act_node_d,
            &na_i,
            &c_i
        )?;
        launch!(
            self,
            act_add,
            na * C,
            &mut *action,
            &*z,
            &act_node_d,
            &na_i,
            &c_i
        )?;
        self.norm(l.norms[LN_ACT], na, true, action)?;
        self.run(l.act_out, action, na, h)?;
        unsafe {
            self.stream.launch_builder(&self.k.policy_offset)
                .arg(&*h).arg(&*facts)
                .arg(&self.b.slice(l.cfg_policy.b..l.cfg_policy.b + D))
                .arg(&act_node_d).arg(&mut *action).arg(&na_i).arg(&d_i)
                .launch_unit(warp_rows(na))
        }.map_err(err)?;
        self.lin(l.policy_action(), h, na, 0.0, input)?;
        const WARPS: u32 = 4;
        let cfg = LaunchConfig {
            grid_dim: (widest.div_ceil(WARPS).max(1), m as u32, 1),
            block_dim: (32, WARPS, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.k.prior)
                .arg(batch.trees.buf())
                .arg(&part_d)
                .arg(&node_d)
                .arg(&row_d)
                .arg(&act_at_d)
                .arg(&cell_at_d)
                .arg(&cells_d)
                .arg(&*input)
                .arg(&*action)
                .arg(&inv_d)
                .arg(&m_i)
                .launch_unit(cfg)
        }
        .map_err(err)
    }

    fn tree(&self, calls: &[Call], mine: &[usize], pack: &mut Pack) -> Res<()> {
        if mine.is_empty() {
            return Ok(());
        }
        for &i in mine {
            let Call::Tree { solve, writes, ncells, nreach, nvals, levels, splits, nterm, seed, .. }
                = &calls[i] else {
                unreachable!("tree shard holds only tree calls")
            };
            let mut b = self.slot(*solve);
            let s = &self.stream;
            let base = pack.words(&writes.blob);
            for r in &writes.runs {
                let dst = b.plan(s, r.dst, r.at as usize, r.len as usize)?;
                pack.piece(dst, r.at, base + r.start, r.len);
            }
            b.level_start.clear();
            b.level_start.extend_from_slice(levels);
            b.level_split.clear();
            b.level_split.extend_from_slice(splits);
            b.nterm = *nterm;
            if let Some(sd) = seed {
                b.seed.put(s, 0, &[*sd])?;
            }
            b.ncells = *ncells;
            b.nreach = *nreach;
            b.reserve(Ent::Cell, *ncells)?;
            b.nvals = *nvals;
            b.reserve(Ent::Reach, (*nreach).max(*nvals))?;
        }
        Ok(())
    }

    fn scatter(&self, pack: &mut Pack) -> Res<()> {
        if pack.moved == 0 {
            return Ok(());
        }
        let moved = pack.moved;
        pack.sum.push(moved);
        let s = &self.stream;
        let mut stage = self.host.lock();
        stage.blob.put(s, pack.blob.len(), copy(&pack.blob))?;
        let np = pack.dst.len();
        let mut i = 0usize;
        while i < np {
            let k = TILE.min(np - i);
            let word0 = pack.sum[i];
            let tile_moved = pack.sum[i + k] - word0;
            let tile_sum: Vec<u32> = pack.sum[i..i + k + 1].iter().map(|x| x - word0).collect();
            stage.at.put(s, k, copy(&pack.at[i..i + k]))?;
            stage.src.put(s, k, copy(&pack.src[i..i + k]))?;
            stage.start.put(s, k + 1, copy(&tile_sum))?;
            stage.dst.put(s, k, copy(&pack.dst[i..i + k]))?;
            let Stage { blob, dst, at, src, start, .. } = &*stage;
            let blob = blob.dev.buf.as_ref().expect("staged");
            let dst = dst.dev.buf.as_ref().expect("staged");
            let at = at.dev.buf.as_ref().expect("staged");
            let src = src.dev.buf.as_ref().expect("staged");
            let start = start.dev.buf.as_ref().expect("staged");
            let (pieces, total_i) = (k as i32, tile_moved as i32);
            launch!(self, scatter, tile_moved as usize, blob, dst, at, src, start, &pieces, &total_i)?;
            i += k;
        }
        Ok(())
    }

}

#[derive(Default)]
struct Stage {
    packed: Wire<u8>,
    xpub: Arr<f32>,
    cards: Wire<f32>,
    card_of_row: Wire<i32>,
    phi: Wire<f32>,
    owner: Wire<u32>,
    pair_part: Wire<u32>,
    pair_at: Wire<u32>,
    pair_board: Wire<u32>,
    pair_config: Wire<u32>,
    pair_start: Wire<u32>,
    board_part: Wire<u32>,
    board_local: Wire<u32>,
    cfg_cards: Wire<f32>,
    blob: Wire<u32>,
    dst: Wire<u64>,
    at: Wire<u32>,
    src: Wire<u32>,
    start: Wire<u32>,
}

#[derive(Default)]
struct Scratch {
    pooled: Arr<f32>,
    h: Arr<f32>,
    z: Arr<f32>,
    input: Arr<f32>,
    leaves: Arr<u32>,
    queries: Arr<f32>,
    public: Arr<f16>,
    kv: Arr<f16>,
    tokens: Arr<f32>,
    projected: Arr<f32>,
    facts: Arr<f32>,
    spatial: Arr<f16>,
    attn_mix: Arr<f16>,
    attn_out: Arr<f16>,
    attn_val: Arr<f16>,
    token_half: Arr<f16>,
    action: Arr<f32>,
}

struct Plan {
    stream: Option<Arc<CudaStream>>,
    bytes: usize,
}

impl Plan {
    fn arr<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Default>(
        &mut self,
        cap: usize,
    ) -> Res<Arr<T>> {
        self.bytes += cap.max(1) * std::mem::size_of::<T>();
        match &self.stream {
            Some(s) => Arr::with_cap(s, cap),
            None => Ok(Arr::default()),
        }
    }

    fn wire<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Default + Copy>(
        &mut self,
        cap: usize,
    ) -> Res<Wire<T>> {
        self.bytes += cap.max(1) * std::mem::size_of::<T>();
        match &self.stream {
            Some(s) => Wire::with_cap(s, cap),
            None => Ok(Wire::default()),
        }
    }

    fn build(mut self, n: usize, slots: usize, cfg: &Cfg) -> Res<(Scratch, Stage, Batch, usize)> {
        let b = &cfg.budget;
        let (nodes, rows, cidx) = (b.cap(Ent::Node), b.cap(Ent::Row), b.cap(Ent::Cidx));
        let cards = slots * NTYPE * TYPE;
        let configs = slots * b.cap(Ent::Config);
        let resident = FIELDS[0] * nodes
            + FIELDS[1] * b.cap(Ent::Cell)
            + FIELDS[2] * b.cap(Ent::Reach)
            + FIELDS[3] * b.cap(Ent::Draw)
            + FIELDS[4] * rows
            + 2 * b.cap(Ent::Config)
            + cidx;
        let scratch = Scratch {
            pooled: self.arr(TILE * C)?,
            h: self.arr(2 * TILE * D)?,
            z: self.arr(TILE * D)?,
            input: self.arr(TILE * (D + 2 * ATTN))?,
            leaves: self.arr(n * cfg.s.max(1) as usize)?,
            queries: self.arr(BOARD_TILE * (N_HEXES + NTYPE) * ATTN)?,
            public: self.arr::<f16>(BOARD_TILE * (N_HEXES + NTYPE) * C)?,
            kv: self.arr::<f16>(BOARD_TILE * (N_HEXES + NTYPE) * 2 * ATTN)?,
            tokens: self.arr(TILE * NSLOT * (3 + TYPE))?,
            projected: self.arr(TILE * NSLOT * ATTN)?,
            facts: self.arr(TILE * D)?,
            spatial: self.arr::<f16>(TILE * N_HEXES * C)?,
            attn_mix: self.arr::<f16>(PAIR_TILE * ATTN)?,
            attn_out: self.arr::<f16>(PAIR_TILE * ATTN)?,
            attn_val: self.arr::<f16>(PAIR_TILE * ATTN)?,
            token_half: self.arr::<f16>(TILE * NTYPE * C)?,
            action: self.arr(TILE * C)?,
        };
        let stage = Stage {
            packed: self.wire(TILE * ROW_BYTES)?,
            xpub: self.arr(TILE * PUBFEAT)?,
            cards: self.wire(cards)?,
            card_of_row: self.wire(TILE)?,
            phi: self.wire(configs * CFEAT)?,
            owner: self.wire(configs)?,
            pair_part: self.wire(PAIR_TILE)?,
            pair_at: self.wire(PAIR_TILE)?,
            pair_start: self.wire(PAIR_TILE + 1)?,
            pair_board: self.wire(PAIR_TILE)?,
            pair_config: self.wire(PAIR_TILE)?,
            board_part: self.wire(BOARD_TILE)?,
            board_local: self.wire(BOARD_TILE)?,
            cfg_cards: self.wire(cards)?,
            blob: self.wire(n * (resident + rows + cidx + 2 * rows + 1))?,
            dst: self.wire(TILE)?,
            at: self.wire(TILE)?,
            src: self.wire(TILE)?,
            start: self.wire(TILE + 1)?,
        };
        let batch = Batch::carve(&mut self, n, slots, cfg)?;
        Ok((scratch, stage, batch, self.bytes))
    }
}

fn err(e: impl std::fmt::Debug) -> String {
    format!("{e:?}")
}
