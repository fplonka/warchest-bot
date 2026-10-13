# gumbel30

Commit 8e117ed, 30 minutes on the rented 2x RTX 3090 box (vast 53124179).
Self-play on GPU 0, training on GPU 1. Seed 1001, snapshots every 7.5 minutes, all other
knobs at the defaults (replay_ratio 1, games 512, batch 1024, cap 1M rows).

The search is Gumbel AlphaZero with 32 simulations per decision. The root samples
m = min(n_legal, 16) moves by Gumbel-top-k and runs Sequential Halving over them
(phases of 16, 8 and 8 simulations). The move is argmax g + logits + sigma(q) among the
most visited survivors, with no temperature. The policy target is
softmax(logits + sigma(completed q)). Nodes below the root select
argmax pi'(a) - N(a) / (1 + sum N). sigma = (50 + max N) * 1.0 * (q + 1) / 2.
There is no Dirichlet noise, PUCT or first-play urgency.
Plots: `uv run --no-project --with matplotlib --with numpy python runs/ismcts-health/plot.py gumbel30 ismcts60b gumbel30`.

## Throughput

| | ismcts60b (PUCT, 256 sims) | gumbel30 (32 sims) | bench, random net, 60 s |
|---|---|---|---|
| evals/s | 439k | 435k | 449k |
| decisions/s | 2380 | 13.8k | 11.0k |
| rows/s | 2222 | 13.8k | 10.4k |
| rows per GPU call | 723 | 464 | 653 |
| GPU 0 busy | 92% | 90% | 91% |
| rows per game | 93 | 76 | |
| buffer span at the end | 446 s | 74 s | |

The GPU stays saturated. Each decision makes four GPU round trips (the root, then one
per Sequential Halving phase), so rows per call fall, but the lanes stay busy.
Self-play makes about six times more rows per second. Evals include the opponent-move
likelihood rows for the belief update, so evals per decision are not 33.
With games=1024 the bench gave 11.6k rows/s (1145 rows per call); with 2048, 9.4k.

## Curve (search phase, 5-minute windows)

| t (s) | timeouts | decisions/game | passes/game | Lv | fresh Lv | value sq err | fresh sq err | Lp | policy KL | target entropy |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-300 | 0.00 | 89 | 7.8 | 1.34 | 1.17 | 0.678 | 0.832 | 1.91 | 0.12 | 1.79 |
| 300-600 | 0.00 | 76 | 2.5 | 1.05 | 1.05 | 0.735 | 0.740 | 1.33 | 0.19 | 1.14 |
| 600-900 | 0.00 | 73 | 1.3 | 1.01 | 1.02 | 0.687 | 0.701 | 0.79 | 0.19 | 0.60 |
| 900-1200 | 0.00 | 75 | 2.0 | 1.00 | 1.02 | 0.665 | 0.684 | 0.65 | 0.19 | 0.45 |
| 1200-1500 | 0.00 | 78 | 2.8 | 0.99 | 1.00 | 0.647 | 0.664 | 0.60 | 0.20 | 0.40 |
| 1500-1800 | 0.00 | 81 | 3.5 | 0.97 | 0.98 | 0.612 | 0.624 | 0.57 | 0.18 | 0.39 |

- No pass explosion and no timeouts at any time (ismcts60b had 79 passes per game and 60%
  timeouts in its first 10 minutes).
- The value head learns without memorizing: train and fresh loss agree, and squared error
  falls from 0.83 to 0.62 (ismcts60b stayed near 0.78).
- The policy target is sharp. Its entropy falls from 2.0 to 0.39, and Lp follows it. With
  c_scale 1.0 and q in [0, 1], a Q gap of 0.1 is worth about 5 logits at 32 simulations.
  This is the paper's Go/chess setting; mctx's default (min-max per node, scale 0.1) is softer.
- The learning rate decays over the last quarter of the run, so the last window is not
  comparable to minute 22-30 of ismcts60b.

## Ladder (packed bots: Gumbel search, sims=32, no temperature)

| match | draft seed | games | W/L/D | score | 95% CI | Elo |
|---|---|---|---|---|---|---|
| s1 (7.5 min) vs greedy.static | 2000 | 200 | 200/0/0 | 1.000 | 1.000-1.000 | +inf |
| s2 (15 min) vs greedy.static | 2000 | 200 | 200/0/0 | 1.000 | 1.000-1.000 | +inf |
| s3 (22.5 min) vs greedy.static | 2000 | 200 | 200/0/0 | 1.000 | 1.000-1.000 | +inf |
| final vs greedy.static | 2000 | 200 | 200/0/0 | 1.000 | 1.000-1.000 | +inf |
| s2 vs s1 | 3000 | 200 | 169/31/0 | 0.845 | 0.795-0.895 | +295 |
| s3 vs s2 | 3000 | 200 | 132/68/0 | 0.660 | 0.601-0.719 | +115 |
| final vs s3 | 3000 | 200 | 108/92/0 | 0.540 | 0.472-0.608 | +28 |
| final vs s1 | 3000 | 200 | 189/11/0 | 0.945 | 0.914-0.976 | +494 |
| final vs ismcts60b.final | 3000 | 200 | 162/38/0 | 0.810 | 0.753-0.867 | +252 |

Raw match lines are in ladder/*.jsonl. The CI uses pairs of games on one draft.
ismcts60b.final keeps its own binary (PUCT, 256 sims, temp 0.5, cut 0.1), so the last row
compares net and search together, and it costs the Gumbel bot an eighth of the simulations.

## Summary

Every snapshot beats the one before it, with smaller gains each time (+295, +115, +28 Elo;
the last is inside the noise and overlaps the learning-rate decay). After 30 minutes the bot
beats the 60-minute PUCT bot in 81% of games while it searches 32 simulations instead of
256. Open questions: whether a softer sigma (c_scale 0.1, or a higher c_visit) keeps the
policy target less peaked and helps later training; whether the 74 s buffer is too short now
that rows come six times faster (raise cap, or raise replay_ratio); and how strength scales
with more simulations at play time.
