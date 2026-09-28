# Self-play throughput log

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
