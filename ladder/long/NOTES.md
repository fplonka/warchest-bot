# The long campaign

Bots trained for 480 min, packed at 1, 2, 4 and 8 h, ranked by
`tools/ladder.py ladder/long report`. The campaign is this directory: `runs`
names the runs in it, one a line, and `matches.jsonl` holds every match
played. The bots come off `runs/<run>/bots/*/bot.json`, which the pull keeps;
the weights stay on the box where the matches run.

The driver is `tools/curve.sh`, which pops a line off `runs/curve/queue`. A
line is a run name, a branch, and any config overrides. It trains, packs,
drops every snapshot but the last, pulls the manifests, adds the run here, and
plays an hour of matches before the next arm.

Every bot plays at s=512, c=8, batch=8, rounds=0 whatever trained it, so the
ladder compares networks and not search budgets.

## 2026-09-14: the first nine arms

master, combined, no-directional, categorical, config-attn, ema, q-005,
deep-12, directional, all at seed 101. no-directional won on the 1 h -> 8 h
gain, but on one seed each, and the 2 h curve put seed noise near 30 Elo, so
the gaps under about 50 Elo in that table are not evidence.
`no-directional-ema` is the same run packed from the raw weights rather than
the moving average; it came out level, so the choice is a wash at 8 h.

## 2026-09-15: the pass before the four week run

Nine arms on seed 1001, which fixes the initial weights and the warm-up
drafts, so every arm starts from the same place. That pairing is why there is
one seed and not two, and the price is that only a gain clearing about 50 Elo
counts as evidence.

The kernels now hold about 178 generated rows a second once the rate settles,
against 120 during the first campaign. Four weeks is then roughly 2.6 billion
optimizer rows against 1.28 M parameters, 6.4 doublings past these arms, and
the 150k replay buffer is a window 14 minutes wide.

- `adam-base` (wd=0) against `adamw`. Weight decay is here because the
  matrices feeding each LayerNorm grow without bound and the step in that
  subspace falls as the square of the norm. Judge it on weight_norm going flat
  first and Elo second.
- `rounds1` grows the tree past the round boundary.
- `policy-005` and `policy-060` bracket policy_w, which moved to 0.18 on a
  gradient-norm argument and has never been played.
- `s768` spends the kernel speedup on target quality rather than volume.
- `deep16` is a sixteen block trunk. It is the one arm whose initial weights
  differ, so it needs a wider margin than the rest.
- `explore-005` halves the uniform noise mixed into the played move. Eight
  hours flatters this knob, because early on the policy is bad anyway.
- `cap2m` widens the replay window from 14 minutes to three hours. It runs
  last because its snapshots are 3.4 GB each and the box has 44 GB free.

A game ends when somebody wins or neither player can act. The harness stops at
256 main plays and a match scores a stopped game as drawn, so most draws in a
report are stopped games.
