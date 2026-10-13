# IS-MCTS against SoG

The bots in `bots` are fourweek.s1 (SoG, 4 h), fourweek.s40 (SoG, 160 h), ismcts4h.s2
(2 h) and ismcts4h.final (4 h). SoG plays at its packed search, s=1024 c=4 DCFR. IS-MCTS
plays at 73728 sims, the budget whose box time per move matches fourweek.s1. Drafts have
no footmen, as in ladder/fourweek. The first session (below) used ismcts60b bots at 256 sims.

## One referee for both protocols

The ismcts ladder speaks protocol 7. Master bots speak protocol 6. Protocol 7 is
protocol 6 without `watch` asks. A v6 bot that is not watched computes its model of the
opponent's node when the opponent's move arrives (`modelled.take()` falls back to
`brain.policy` in master's `Session::acted`), with the same state, beliefs and rng. So it
plays the same game; the opponent-model search only moves from the opponent's turn into
its own. The game rules are the same: `units`, `actions` and `board` do not differ, `rules`
differs only in where the play counter lives, PLAY_LIMIT is 256 on both, `obs_key` is the
same, and both builds report rules hash 8544657076115779065. The only change is that
`ladder` accepts protocol 6 in the greeting.

## Where the bots live

`tools/ladder_box.sh` refreshes the bots the training box (ssh2.vast.ai:17198, instance
50497199) holds into `runs/` here, read-only, and pushes the local copies to the rented
box. Bots from other boxes must be copied into `runs/` first; ismcts60b's came from
instance 53124179.

The rented box gets the bots from this machine's uplink. For four bots that took about
20 minutes, before the session clock starts.

## 2026-09-28: the first session

22 matches of 100 games on one 2x RTX 3090 box (vast 53148847), two hours. Every game
passed the strict rules-hash check. The CI uses pairs of games on one draft.

| bot | trained | lr at the snapshot |
|---|---|---|
| ismcts60b.s2 | 30 min | 1e-3, constant |
| ismcts60b.final | 60 min | 1e-4, after a cosine decay over the last 15 min |
| fourweek.s1 | 4 h | 1e-3, constant |
| fourweek.s40 | 160 h | 2.6e-4, on the cosine 3e-4 -> 3e-5 that began at 80 h |

| match | games | W/L/D | score | 95% CI | Elo |
|---|---|---|---|---|---|
| fourweek.s1 vs ismcts60b.final | 900 | 831/69/0 | 0.923 | 0.907-0.940 | +432 |
| fourweek.s40 vs ismcts60b.final | 100 | 99/1/0 | 0.990 | 0.970-1.010 | +798 |
| fourweek.s40 vs ismcts60b.s2 | 100 | 97/3/0 | 0.970 | 0.937-1.003 | +604 |
| fourweek.s40 vs fourweek.s1 | 500 | 396/104/0 | 0.792 | 0.759-0.825 | +232 |
| ismcts60b.final vs ismcts60b.s2 | 600 | 529/71/0 | 0.882 | 0.856-0.907 | +349 |

The fit (`tools/ladder.py ladder/ismcts report`): ismcts60b.s2 -527, ismcts60b.final -186,
fourweek.s1 +242, fourweek.s40 +471. The 4 h SoG bot is about 430 Elo above the 1 h
IS-MCTS bot. Search budgets differ a lot (SoG 1024 solves with c=4 against 256 IS-MCTS
sims), and thinking time per move was not measured.

## 2026-09-29: ismcts4h at matched time

Box time per move: each bot plays itself, 100 games (the campaign's concurrency), one bot
process per GPU, on the idle box (vast 53191824). Time per move = match wall time / moves.

| bot | sims | box time per move |
|---|---|---|
| fourweek.s1 | SoG s=1024 c=4 | 53.3 ms |
| fourweek.s40 | SoG s=1024 c=4 | 53.7 ms |
| ismcts4h.final | 4096 | 3.5 ms |
| ismcts4h.final | 32768 | 22.8 ms |
| ismcts4h.final | 65536 | 47.9 ms |
| ismcts4h.final | 73728 | 53.1 ms |
| ismcts4h.s2 | 73728 | 56.0 ms |

Time grows a little faster than linearly in sims. The matched budget is 73728 sims, about
3x the prior guess of 25k.

Session: 14 matches of 100 games in 2 hours on the same box, all with the strict rules-hash
check. `matches.jsonl` rows now carry `moves`.

| bot | trained | lr | sims | time per move |
|---|---|---|---|---|
| ismcts4h.s2 | 2 h | 2.8e-3, constant | 73728 | 56.0 ms |
| ismcts4h.final | 4 h | 2.8e-3, constant | 73728 | 53.1 ms |
| fourweek.s1 | 4 h | 1e-3, constant | SoG s=1024 c=4 | 53.3 ms |
| fourweek.s40 | 160 h | 2.6e-4, on a cosine 3e-4 -> 3e-5 | SoG s=1024 c=4 | 53.7 ms |

| match | games | W/L/D | score | 95% CI | Elo |
|---|---|---|---|---|---|
| ismcts4h.final vs ismcts4h.s2 | 300 | 158/142/0 | 0.527 | 0.470-0.583 | +19 |
| fourweek.s1 vs ismcts4h.s2 | 300 | 53/247/0 | 0.177 | 0.134-0.220 | -267 |
| fourweek.s1 vs ismcts4h.final | 200 | 35/165/0 | 0.175 | 0.122-0.228 | -269 |
| fourweek.s40 vs ismcts4h.s2 | 300 | 124/173/3 | 0.418 | 0.363-0.474 | -57 |
| fourweek.s40 vs ismcts4h.final | 300 | 120/180/0 | 0.400 | 0.345-0.455 | -70 |

The fit (`tools/ladder.py ladder/ismcts report`, with the older s1-vs-s40 games):
fourweek.s1 -197, fourweek.s40 +26, ismcts4h.s2 +78, ismcts4h.final +92 (+-15 on the
branch gains). At equal box time per move, the 4 h IS-MCTS bot is about 290 Elo above the
4 h SoG bot and about 65 Elo above the 160 h SoG bot. Training from 2 h to 4 h at a
constant lr adds little (+14 to +19 Elo, inside the noise).
