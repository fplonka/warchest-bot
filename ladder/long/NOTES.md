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

Every bot plays at the deployment budget, s=1024, c=8, batch=16, rounds=1,
whatever trained it, so the ladder compares networks and not search budgets.
A match takes two to three times as long as at s=512, so the driver plays
three hours of matches after each arm.

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

## 2026-09-17: restart on the corrected drafts

The 136 matches of the first campaign and the 28 that followed were played by
a ladder binary built before the draft dealer was fixed on 2026-09-13, so the
harness dealt from the old 19-card pool. All of them are gone, and the runs
list is cut to strength anchors: adam-base and adamw (the current recipe at
seed 1001), no-directional-ema and combined (the same recipe on the s=512
kernels), master from before the combined recipe, and pre-changes_s1 as the
floor. `defaults` is the first arm on the corrected ladder: round_batch=16
against adamw's 8, otherwise the same run.

## 2026-09-19: the search budget, one net at a time

The ladder was stopped after 71 matches (8 h column: adam-base +285,
adamw +281, s256-c1 +280, defaults +232, s256-c05 +97, all +-25). At matched
optimizer rows every rounds=1 arm is the same bot, so the 8 h ranking is a
ranking of rows a second, and it cannot say what a deeper target is worth
once the net is no longer data starved.

`search.jsonl` answers that directly: one fixed net, played against itself
under different searches, 200 games a match, both seat orders. Elo relative
to the training search (256 expansions, 256 iterations) on the s256-c1 8 h
net:

| expansions | iterations | Elo | cost |
|---|---|---|---|
| 1024 | 128 | -36 | 2x (the old deployment search) |
| 64 | 256 | -104 | 0.25x |
| 128 | 256 | +10 | 0.5x |
| 512 | 256 | +24 | 2x |
| 1024 | 256 | +60 | 4x |
| 128 | 512 | -45 | 1x |
| 256 | 512 | -17 | 2x |
| 512 | 512 | +56 | 4x |
| 256 | 1024 | +7 | 4x |

128 to 256 iterations is worth 50 to 100 Elo on every net tried (s256-c1 at
1 h and 8 h, adam-base, defaults), 256 to 1024 is worth nothing, and
expansions pay about 25 Elo a doubling of cost after that. The big tree's
edge over (256, 256) fell from ~120 Elo on the 1 h net to ~57 on the 8 h
net, so a better net wants the extra tree less, not more. The four week run
therefore trains at s=256, c=1 throughout, and bots play at (1024, 4).
Rerun this grid on the week 1 and week 2 snapshots; a growing edge for the
big tree is the only signal that would justify switching mid-run.
