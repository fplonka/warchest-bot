# ismcts60

Commit 60e269a, 60 minutes on one rented 2x RTX 3090 box (vast 53027623, 128 cores).
Self-play on GPU 0, training on GPU 1. Seed 1001, the new default knobs: sims=256,
Dirichlet root noise (0.25, alpha 10/n), self-play temp=0.5, target_every=1 min,
uniform replay sampling, games=512, batch=1024, replay_ratio=6. Snapshots every 15 minutes.

## Throughput

| | ismcts30 (91d412b, sims=128) | ismcts60 (60e269a, sims=256) |
|---|---|---|
| evals/s | 364k | 443k |
| decisions/s | 2625 | 2034 |
| rows/s | 2472 | 1975 |
| rows per GPU call | 932 | 826 |
| trained rows/s | 14.9k | 11.8k |
| optimizer rows / generated rows | 6.00 | 6.00 |
| GPU 0 busy | 94% | 93% |
| rows per game | 172 | 122 |

The memo and trunk work give +22% evals/s. Doubled sims halve decisions per eval,
so rows/s drops by only 20%. Games are shorter because far fewer hit the play limit.
The trainer keeps replay ratio 6 with headroom.

## Curve (search phase, 10-minute windows)

| t (s) | timeouts | Lv | Lp | policy KL | target entropy |
|---|---|---|---|---|---|
| 0-600 | 0.02 | 0.98 | 1.89 | 0.11 | 1.79 |
| 600-1200 | 0.01 | 0.61 | 2.11 | 0.32 | 1.79 |
| 1200-1800 | 0.01 | 0.55 | 2.08 | 0.32 | 1.76 |
| 1800-2400 | 0.03 | 0.55 | 2.06 | 0.23 | 1.83 |
| 2400-3000 | 0.04 | 0.57 | 2.08 | 0.19 | 1.90 |
| 3000-3600 | 0.04 | 0.57 | 2.11 | 0.16 | 1.95 |

The timeout rate is 0.01-0.04 (ismcts30: 0.10-0.21). Lv is lower than in ismcts30
(0.57 against 0.70). Lp still stays near 2.1, and after 30 minutes the target
entropy rises and the KL falls again.

## Ladder (packed bots: sims=256, temp=0.5, cut=0.1)

| match | draft seed | games | W/L/D | score | 95% CI | Elo |
|---|---|---|---|---|---|---|
| final vs greedy.static | 1000 | 400 | 349/39/12 | 0.887 | 0.856-0.919 | +359 |
| final vs greedy.static | 2000 | 200 | 168/26/6 | 0.855 | 0.809-0.901 | +308 |
| s1 (15 min) vs greedy.static | 1000 | 400 | 393/5/2 | 0.985 | 0.974-0.996 | +727 |
| s1 (15 min) vs greedy.static | 2000 | 200 | 197/3/0 | 0.985 | 0.968-1.002 | +727 |
| s2 (30 min) vs greedy.static | 2000 | 200 | 180/16/4 | 0.910 | 0.873-0.947 | +402 |
| s3 (45 min) vs greedy.static | 2000 | 200 | 176/16/8 | 0.900 | 0.862-0.938 | +382 |
| s2 vs s1 | 3000 | 200 | 71/129/0 | 0.355 | 0.289-0.421 | -104 |
| s3 vs s1 | 3000 | 200 | 61/138/1 | 0.307 | 0.246-0.369 | -141 |
| final vs s1 | 3000 | 400 | 116/283/1 | 0.291 | 0.245-0.338 | -154 |

Raw match lines are in ladder/*.jsonl. The CI uses pairs of games on one draft.
The ismcts30 final bot could not be laddered: its weights were only on the destroyed
box. On the same draft seed, final scores 0.887 against greedy where ismcts30 final
scored 0.720.

The new defaults are much stronger, but strength peaks at 15 minutes and then falls:
every later snapshot loses clearly to s1. The drop comes while Lv is flat and while
the learning rate is still constant, so the next question is what self-play drifts
towards after 15 minutes, not the schedule.
