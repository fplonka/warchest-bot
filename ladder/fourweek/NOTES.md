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
- `tools/ladder.py ladder/fourweek report` ranks the bots in `runs` here,
  against s256-c1 and adam-base from the long campaign as anchors (+280 and
  +285 at 8 h on the old scale, all re-manifested to the (1024, 4) search).

## Playing the ladder

Never on the training box. On a second box:

    export WARCHEST_BOX_HOST=<host> WARCHEST_BOX_PORT=<port>
    tools/box.sh setup && tools/box.sh sync && tools/box.sh build
    WARCHEST_BOX_DIR=/workspace/curve tools/box.sh sync && tools/box.sh build

then copy `runs/fourweek/bots`, `runs/s256-c1/bots` and `runs/adam-base/bots`
from the training box into `/workspace/warchest-engine/runs/` on it, and run
`tools/ladder.sh ladder/fourweek <hours>` with the same env vars. It pairs
bots of equal age across runs and appends to `matches.jsonl` here.
