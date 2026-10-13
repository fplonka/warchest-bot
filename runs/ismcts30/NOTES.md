# ismcts30

Commit 91d412b, 30 minutes on one rented 2x RTX 3090 box (vast 53030355, 192 cores).
Self-play on GPU 0, training on GPU 1. Seed 1001, default knobs: sims=128,
games=512, batch=1024, replay_ratio=6. Snapshots every 10 minutes.

## Throughput

| | before (3ed8168) | after (91d412b) |
|---|---|---|
| self-play GPUs | 0 and 1 (shared with training) | 0 |
| evals/s | 692k (bench, no trainer) | 364k |
| rows per GPU call | 225 | 930 |
| trained rows/s | 2.5k, one fit call blocked 260 s | 16k |
| optimizer rows / generated rows | 0.71 (target 6) | 6.00 |

The trunk kernel runs at about 164 MFLOP per row, which is near the 3090 fp16
tensor peak, so evals/s per GPU is flat in batch size. The belief filter uses
13% of evals. In self-play, GPU 0 is 98% busy and GPU 1 80%. CPU use is below 20%.

## Curve (search phase, 5-minute windows)

| t (s) | timeouts | Lv | Lp | policy KL |
|---|---|---|---|---|
| 0-300 | 0.16 | 1.04 | 2.12 | 0.04 |
| 300-600 | 0.14 | 0.82 | 2.16 | 0.06 |
| 600-900 | 0.10 | 0.67 | 2.20 | 0.19 |
| 900-1200 | 0.13 | 0.65 | 2.19 | 0.21 |
| 1200-1500 | 0.16 | 0.67 | 2.18 | 0.18 |
| 1500-1800 | 0.21 | 0.70 | 2.19 | 0.14 |

## Ladder vs greedy.static (packed bots: sims=256, temp=0.5, cut=0.1)

| bot | games | W/L/D | score | 95% CI | Elo |
|---|---|---|---|---|---|
| s1 (10 min) | 200 | 152/41/7 | 0.777 | 0.723-0.832 | +217 |
| s2 (20 min) | 200 | 148/39/13 | 0.772 | 0.714-0.831 | +212 |
| final (30 min) | 400 | 279/103/18 | 0.720 | 0.677-0.763 | +164 |

Raw match lines are in ladder/*.jsonl. The CI uses pairs of games on one draft.
Strength does not grow after 10 minutes, and the timeout rate rises again late in the run.
