# Self-play throughput log

## Result

| | decisions/s | evals/s | evals/decision |
|---|---|---|---|
| before (7519131) | 27.96k | 926k | 33.1 |
| after (cb2caaf2) | 32.45k | 1079k | 33.3 |
| change | +16.1% | +16.5% | |

What limits throughput now: both GPUs run at the 330 W host power cap, SMs are full of trunk CTAs,
and the trunk's cost is spread over many non-MMA phases (see the ablations below). The trainer on
GPU 1 gets ~5 steps/s against the ~32/s that replay_ratio=1 asks for.

Box: vast 53191824, 2x RTX 3090 (330 W host power cap, SM clocks ~1650 MHz under load),
Threadripper PRO 3975WX (64 threads). The first box went down; this one replaced it.

Protocol: `python train/train.py out=<name> init_weights=runs/perf-ref/snap_0004.pt minutes=4 warm_minutes=0`.
perf-ref is a 25-minute baseline run (~83 decisions per game). The result is the mean of the
`decisions_per_s` and `evals_per_s` reports from t >= 60 s (18 reports). Two baseline repeats
agree within 0.2%.

## Accounting of the baseline (7519131)

- 27.96k decisions/s, 926k evals/s, 33.1 GPU evals per decision.
- Rows asked of the memo per decision: 31.6 leaf, 6.1 belief (posterior), 1 root. The memo
  answers 15.5% of them. An 8x larger memo does not raise the hit rate.
- Game threads: ~1000 of 1024 wait on the GPU at any time. CPU is ~35% busy. The 10 driver
  threads spend 91% of their time inside `Card::eval`. The system is GPU-bound.
- nsys, 12 s of a real run: both GPUs are 97-99% busy (union of kernel time). k_trunk runs
  ~460k boards/s per GPU. Median trunk grid is 31 blocks (93 rows) on GPU 0, 170 on GPU 1.
  Side kernels (3 fp32 cuBLAS gemms, 2 bias, value, actions, prior, expand) sum to ~35% of
  the span on GPU 0, overlapped with trunk blocks from other lanes.
- A trunk kernel alone (big batch, one stream) runs 370 us per wave of 82 blocks, i.e.
  615k boards/s per GPU. The real pipeline gets 75% of that.
- Without the head kernels (uniform priors, zero values; changes the games) the GPUs give
  1.11M evals/s against 926k. The side kernels cost ~20% of GPU throughput.
- Trainer: ~7 steps/s of 1024 rows, far below the 28k rows/s that self-play makes. The
  replay ratio is not held.

## Changes

| change | decisions/s | evals/s | evals/decision | note |
|---|---|---|---|---|
| baseline 7519131 | 27.96k | 926k | 33.1 | |
| 3 lanes per GPU | 26.03k | 858k | 32.9 | rejected |
| 8 lanes per GPU | 24.85k | 814k | 32.8 | rejected |
| memo 8x larger | 27.69k | 914k | 33.0 | rejected |
| one queue for all lanes | 29.09k | 958k | 32.9 | kept, +4% |
| heads fused into the trunk epilogue (fp32, 3 boards per CTA) | 29.09k | 972k | 33.4 | rejected: CTA 370 -> 434 us per wave |
| one tensor-core head kernel per 16 rows, no cuBLAS | 31.50k | 1046k | 33.2 | kept, +8% |
| head kernel: branch-free action loads, one thread per value bin | 31.77k | 1061k | 33.4 | kept, +0.9% |
| expand_rows folded into the trunk stem (features from row bytes) | 31.00k | 1031k | 33.3 | rejected: the stem slows more than the kernel saves |
| stem reads tokens from a host-projected table (cards, seats, piles through tok_stem) | 32.45k | 1079k | 33.3 | kept, +2.1% |
| LN partials moved behind the neighbour rows, 2 syncs per layer fewer | 32.48k | 1082k | 33.3 | rejected: +0.1%, power-capped |

Why the side kernels cost so much: k_trunk uses 168 registers x 384 threads, the whole register
file, so nothing co-resides with a trunk CTA. Every block of a side kernel holds an SM that a
trunk CTA then waits for. Fusing the heads into the trunk epilogue removes that, but each CTA then
streams 1.26 MB of fp32 head weights for 3 boards (clock64 per phase: board_out 55k cycles,
hidden/act 31k, against 52k per trunk layer). One batched head kernel reads the fp16 weights once
per 16 rows, with mma, and replaces 3 cuBLAS gemms and 5 small kernels.

Belief rows that need no eval (one legal action, or one matching world and move): 3% of belief
rows, 0.5% of all evals. Not worth code.

k_heads per 16-row block (clock64, test positions): board gemm 22k cycles, mid gemm 10k, value
23k -> 18k, actions 93k -> 53k (~150 actions). The whole kernel is now ~3% of the trunk's SM time.

k_trunk per CTA (3 boards, clock64): stem 77k cycles, 12 layers of ~52k, total ~734k. In a layer:
LN1+store 5.4k, pool_rows 1.9k, pool_gemm+gemm<2> 23k, store 1.3k, gather 3.4k, LN2+store 6.2k,
gemm<1> ~10k. The gemms run at ~80% of the tensor pipe; LN, pooling and stores are ~30% of a layer.

Power: both GPUs sit at the 330 W cap (SW power cap active 100% of samples, 1635-1680 MHz, memory
at 9501 MHz). Throughput is energy per eval, so cutting idle cycles pays only in part.

Single-GPU energy bench (standalone evals of 1.2M distinct positions, 10 threads x 500 rows, power
capped at 328 W). HEAD: 600k and 597k evals/s. Ablations, results wrong but work shape kept:
- main gemm mma skipped: 621k (+3.5%), clock rises to 1860 MHz. The MMAs are not the bottleneck.
- 6/7 of A-fragment ldmatrix skipped: 608k (+1.5%).
- weight B fragments from 2 k-tiles only (L1-hot): 625k (+4%). L2 weight streaming costs ~4%.
- gather mma skipped: 597k (0%).
- LN statistics skipped (syncs kept): 658k (+9.7%).
- per-layer pooling (pool_rows + pool_gemm) skipped: 706k (+18%).
Unrolling pool_gemm's weight loads fully: 544k (worse, register pressure); by 3: 599k (same).
The pooling and LN phases are where the next trunk gains are; both need a rework of how a CTA
reduces across its 12 warps, which is out of scope here.

# Trainer throughput log

## Result

The measure is optimizer rows per second in a real `train.py` run, with optimizer rows equal to
generated rows. Protocol as above: `init_weights=runs/perf-ref/snap_0004.pt warm_minutes=0`,
4-6 minutes, rates over the reports from t >= 90 s.

| | trained rows/s | generated rows/s | trained/generated | GPU split |
|---|---|---|---|---|
| before (ddc3475, batch 1024) | 5.4k | 31.8k | 0.15 | self-play on both GPUs, trainer on GPU 1 at equal priority |
| after (c4194d5, batch 8192) | 25.3k | 25.3k | 1.000 | self-play on both GPUs, trainer on GPU 1 at high priority |
| change | 4.7x | | | |

Two runs at the final defaults gave 25.30k and 25.31k.

## Changes, one at a time

| change | trained rows/s | note |
|---|---|---|
| baseline ddc3475 | 5.4k | ratio 0.15 |
| backpressure + high-priority training stream, chunk 16384 | 19.7k | ratio 1.000; kept |
| batch 4096, lr 2e-3 (sqrt scaling), EMA per row | 22.2k | kept |
| TORCHINDUCTOR_MAX_AUTOTUNE, batch 4096 | 23.1k | +4%, but +40 s compile and NaN losses on a second try; rejected |
| chunk 65536 (probe forward is 1/16 as frequent) | 23.0k | +3%, kept |
| hex-major trunk (adjacency as one GEMM, not bmm) | same | idle-GPU bench 39.1k vs 40.0k rows/s; rejected |
| batch 8192, lr 2.8e-3 | 25.3k | kept |

Backpressure: every game waits before a decision while the farm holds a full chunk of untaken
rows. `Farm.collect` waits for a full chunk. The trainer takes a chunk only after it has trained
all of its debt, so the ratio holds by construction (0.9998 over the run).

## Where the time goes

Idle GPU 1, batch 1024: 25.7 ms per step (40k rows/s). The GEMMs are 39% of GPU time; the rest
is inductor's fused LayerNorm, GELU and cast kernels, which are memory bound on [B, 37, 192]
activations with an fp32 residual. Clip, AdamW and EMA are 4.3 ms per step, which batch 8192
amortizes. Gather, mirror and host copies overlap the GPU and are not on the critical path.

In a live run (torch profiler inside `train.py`, batch 4096) a step takes 195 ms against 92 ms on
an idle GPU. The CPU spends its time in "Command Buffer Full": the trainer is GPU bound. Its
kernels wait for SMs that self-play trunk CTAs (1.2 ms each, the whole register file) hold, and a
high stream priority only orders pending blocks. Fewer, larger kernels per row wait less, which
is why batch 8192 helps more on the shared GPU than on an idle one (idle: 1024 -> 4096 is +12%).

Balance: self-play makes ~16k rows/s per whole GPU and the trainer ~45k rows/s on a whole GPU.
With backpressure the split on GPU 1 settles by itself. The ceiling with an infinitely fast
trainer is ~32k rows/s; at 25.3k the rest is GPU 1's share for the trainer.

Loss check at equal settings (bench_train, batch 1024, lr 1e-3, 200 steps from warm rows): Lv
1.8246 / Lp 2.0499 after, against 1.8022-1.8116 / 2.0420-2.0465 over three runs before. The
spread is run-to-run nondeterminism; the step is numerically the same code.

Replay buffer: 4M rows, about 2.7 minutes of play at 25k rows/s, ~2.6 GB of RAM and a ~2 GB
snapshot.
