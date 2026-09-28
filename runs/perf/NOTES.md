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
