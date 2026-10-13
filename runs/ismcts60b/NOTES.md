# ismcts60b

Commit 9f8de98, 60 minutes on one rented 2x RTX 3090 box (vast 53124179).
Self-play on GPU 0, training on GPU 1. Seed 1001 and the ismcts60 knobs, with two changes:
replay_ratio 6 -> 1, and first-play urgency: an unvisited edge starts at the mover's mean Q
at that node minus 0.25, not at 0. Snapshots every 15 minutes. The trainer now also logs
value and policy loss on each batch of fresh rows before it trains on them (`fresh_*`).
Plots: `uv run --no-project --with matplotlib --with numpy python runs/ismcts-health/plot.py ismcts60b ismcts60 ismcts60b`.

The first launch gave NaN from the first optimizer step. The cause was a stale
torchinductor/triton cache on the box that broke the compiled loss on GPU 1 only (eager
was fine, and GPU 0 was fine). The earlier aborted h_rr2/t2 runs hit the same NaN. After
the caches were deleted, the run was clean. The box ssh port was down for part of the
run. Training did not stop.

## Throughput

| | ismcts60 | ismcts60b |
|---|---|---|
| evals/s | 444k | 439k |
| decisions/s | 2034 | 2380 |
| rows/s | 1975 | 2222 |
| rows per GPU call | 845 | 723 |
| trained rows/s | 11.8k | 2.2k |
| optimizer rows / generated rows | 6.00 | 1.00 |
| GPU 0 busy | 93% | 92% |
| rows per game | 122 | 93 |
| buffer span at the end | 522 s | 446 s |

Self-play speed is the same. The trainer does a sixth of the work.

## Curve (search phase, 10-minute windows)

| t (s) | timeouts | decisions/game | passes/game | Lv | fresh Lv | value sq err | fresh sq err | Lp | policy KL | target entropy |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-600 | 0.60 | 200 | 79.4 | 1.60 | 1.58 | 0.336 | 0.235 | 1.72 | 0.06 | 1.66 |
| 600-1200 | 0.08 | 116 | 18.0 | 1.45 | 1.37 | 0.478 | 0.720 | 1.90 | 0.06 | 1.85 |
| 1200-1800 | 0.00 | 89 | 5.9 | 1.10 | 1.09 | 0.780 | 0.805 | 2.10 | 0.08 | 2.01 |
| 1800-2400 | 0.00 | 83 | 4.2 | 1.08 | 1.09 | 0.783 | 0.802 | 2.01 | 0.10 | 1.91 |
| 2400-3000 | 0.00 | 81 | 3.5 | 1.08 | 1.08 | 0.783 | 0.796 | 1.94 | 0.11 | 1.83 |
| 3000-3600 | 0.00 | 81 | 3.2 | 1.07 | 1.08 | 0.771 | 0.782 | 1.86 | 0.11 | 1.75 |

The same windows for ismcts60: passes/game 10 -> 29, decisions/game 105 -> 141,
value sq err 0.285 -> 0.015, target entropy 1.79 -> 1.95.

- The first 15 minutes are bad: passes go to about 90 per game, claim_initiative to 28,
  and 70% of games hit the play limit. This goes away between 10 and 18 minutes.
  After that, games get shorter all hour (81 decisions, 3 passes, no timeouts) and
  attacks and deploys rise. This is the opposite of the ismcts60 drift.
- The value head no longer memorizes. Train and fresh loss are the same from 20 minutes
  on (1.07 against 1.08). But the value head also learns little: squared error stays near
  0.78, and 1.0 is what a constant 0 gets on decided games. ismcts60 fit its training rows
  to 0.015. (It logged no fresh loss.)
- Lp falls from 2.10 to 1.86 after 20 minutes, and target entropy falls from 2.04 to 1.73.
  In ismcts60 both rose over the same period. The policy KL stays low (0.06-0.11 against 0.16-0.37).

## Ladder (packed bots: sims=256, temp=0.5, cut=0.1)

| match | draft seed | games | W/L/D | score | 95% CI | Elo |
|---|---|---|---|---|---|---|
| s1 (15 min) vs greedy.static | 2000 | 200 | 192/4/4 | 0.970 | 0.949-0.991 | +604 |
| s2 (30 min) vs greedy.static | 2000 | 200 | 200/0/0 | 1.000 | 1.000-1.000 | +inf |
| s3 (45 min) vs greedy.static | 2000 | 200 | 200/0/0 | 1.000 | 1.000-1.000 | +inf |
| final vs greedy.static | 2000 | 200 | 200/0/0 | 1.000 | 1.000-1.000 | +inf |
| s2 vs s1 | 3000 | 200 | 180/20/0 | 0.900 | 0.858-0.942 | +382 |
| s3 vs s2 | 3000 | 200 | 146/54/0 | 0.730 | 0.670-0.790 | +173 |
| final vs s3 | 3000 | 200 | 125/75/0 | 0.625 | 0.561-0.689 | +89 |
| final vs s1 | 3000 | 200 | 198/2/0 | 0.990 | 0.976-1.004 | +798 |
| final vs ismcts60.s1 | 3000 | 200 | 199/1/0 | 0.995 | 0.985-1.005 | +920 |

Raw match lines are in ladder/*.jsonl. The CI uses pairs of games on one draft.
The ismcts60b bots search with the new first-play urgency. ismcts60.s1 keeps its own
binary (Q=0 for unvisited edges). So the last row compares net and search together.

## Compared with ismcts60

ismcts60 peaked at 15 minutes, and every later snapshot lost to s1 (final vs s1: 0.291).
In ismcts60b every snapshot beats the one before it, and the gain per 15 minutes gets
smaller (+382, +173, +89 Elo). From 30 minutes on it beats greedy.static in all 200 games,
and the final bot beats ismcts60's best bot in 199 of 200 games. Strength now keeps
rising over the hour.

The run does not show which change did this. Both went in together. The pass explosion in
the first 15 minutes is new and matches first-play urgency on an untrained value head:
the search tries the top-prior move and rarely leaves it. The value head is now honest
but weak (squared error 0.78). The next questions are a single-change run for each knob,
and whether more optimizer work per row (ratio 2-3) makes the value head useful without
memorizing, with the fresh loss as the check.
