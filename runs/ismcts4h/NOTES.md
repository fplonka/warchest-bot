# ismcts4h

Four hours from scratch at c4194d5 with the phase-1 trainer defaults: batch 8192, lr 2.8e-3
constant (`lr_stable_frac=1.0`), buffer 4M rows, replay ratio 1, 32 sims in self-play.
Command: `tools/box.sh go out=ismcts4h minutes=240 lr_stable_frac=1.0 snapshot_every=60`.
Box: vast 53191824, 2x RTX 3090.

Health: every loss is finite. Optimizer rows / generated rows = 0.99999 (376.68M / 376.68M).
Play-limit timeouts were 17% during the first minutes after the warm phase and fell to 0 by
minute 8; they stay under 0.05% after that. Throughput after minute 10: 26.3k rows/s.

| t | Lv | Lp | fresh Lv | fresh Lp | decisions/game | rows/s |
|---|---|---|---|---|---|---|
| 15 min | 1.038 | 1.184 | 1.036 | 1.087 | 73.7 | 24.6k |
| 30 min | 0.978 | 0.626 | 0.982 | 0.630 | 79.6 | 26.1k |
| 1 h | 0.917 | 0.582 | 0.926 | 0.589 | 88.8 | 26.3k |
| 2 h | 0.893 | 0.586 | 0.903 | 0.591 | 92.3 | 26.0k |
| 3 h | 0.884 | 0.592 | 0.896 | 0.599 | 92.5 | 26.0k |
| 4 h | 0.879 | 0.597 | 0.891 | 0.603 | 92.5 | 26.3k |

Means over +-5 minutes. The value loss still falls slowly at 4 h; the policy loss is flat after
1 h, which at a constant lr and a moving target is expected.
