# bigref

Ten minutes from scratch on the 34M net (C=512, 24 blocks, D=1024) at f7fa945, 1024 games.
Command: `python train/train.py out=bigref minutes=10`. Box: vast 50497199, 2x RTX 3090.

Net: 34.0M params, 1.43 GFLOP per eval (trunk 1.42: per block 2·37·(2C·C + C·C) + 2·2C·C).

Health: every loss is finite. Lv falls from 2.77 (first warm step) to 1.85, Lp stays near 2.1.
Optimizer rows / generated rows = 0.996 (chunk granularity). Play-limit timeouts are 77% at
5 minutes and 67% at 10 minutes: the net is still near random, so this run does not show them
vanish.

## Throughput

Measured from this run's 10-minute snapshot: `train.py init_weights=runs/bigref/snap_0001.pt
minutes=8 warm_minutes=0`, rates over t >= 150 s. Busy is nvidia-smi utilization.

| | trained rows/s | evals/s | gpu0 busy | gpu1 busy |
|---|---|---|---|---|
| old 2.7M net (90686e3, from scratch) | 15.7k | 605k | 65% | 81% |
| f7fa945, 1024 games, 8 trunk warps | 2241 | 78.8k | 99% | 99% |
| 0805a69, 2048 games, 16 trunk warps | 2588 | 79.6k | 83% | 87% |

Self-play alone (tools/bench.py, same snapshot) gives 103k evals/s at 0805a69. Both GPUs sit
at their power limit (324 W and 292 W, 1.3-1.6 GHz). At 0805a69 the GPUs idle 13-17%: games
wait on backpressure while the trainer, which shares GPU 1, finishes its chunk. Giving the
trainer more of GPU 1 is the next lever.
