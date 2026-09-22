# The four week run

`runs/fourweek` trains at the config.py defaults (s=256, c=1, cap 1M,
explore 0.03) for 40320 minutes. It packs a bot every four hours into
`runs/fourweek/bots/s<n>`, keeps the last two full checkpoints, and
resumes itself after a crash (`tools/resume_train.sh`).

## Checking on it

- `tools/box.sh pull fourweek` copies the manifests, `log.json`, `epochs.jsonl`
  and `train.log` here; the weights stay on the box.
- `tools/monitor.py` reads `runs/fourweek/epochs.jsonl`: rows a second, loss,
  weight norm, policy_search_kl.
- `tools/ladder.py ladder/fourweek report` ranks the bots listed in `bots`
  here (one glob a line), against s256-c1 and adam-base from the long
  campaign as anchors (+280 and +285 at 8 h on the old scale). The fourweek
  bots are log-spaced in age, 100 games a match, at s=256 rather than the
  packed 1024: cheap enough to connect the ladder in one session.
- `fourweek-lr3` branches s17 (68 h) at lr 3e-4 held constant for 8 h, to
  test whether the run is learning-rate limited: compare its 76 h bot with
  fourweek's.

## Playing the ladder

Never on the training box. `tools/fourweek_ladder.sh <hours>` rents a 2x3090
AMD box (the packed bots are built for the trainer's Zen CPU), builds the
ladder binary, copies the bots over from the training box, plays the most
informative pairs for that long, appends to `matches.jsonl` here, prints the
report and destroys the box. Every session continues the same campaign.
