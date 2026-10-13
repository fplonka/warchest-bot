# ismcts health: why strength peaks early

Plots come from `plot.py` over runs/ismcts{30,60}/epochs.jsonl. Snapshots: ismcts30 at 10/20/30 min, ismcts60 at 15/30/45/60 min.
(`uv run --no-project --with matplotlib --with numpy python runs/ismcts-health/plot.py ismcts-health ismcts30 ismcts60`).
Both runs show the same shape. ismcts30 predates the memo and the new defaults, and its
first snapshot (10 min) is also its strongest. So the memo and the four default changes
did not cause the drop. They only moved it.

| run | window (min) | decisions/game | passes/game | maneuvers/game | timeouts | value sq err |
|---|---|---|---|---|---|---|
| ismcts30 | 3.5-6 | 177 | 34.3 | 72.0 | 0.155 | 0.235 |
| ismcts30 | 8-11 (peak) | 169 | 30.9 | 69.0 | 0.101 | 0.075 |
| ismcts30 | 27-30 | 198 | 41.3 | 81.5 | 0.209 | 0.016 |
| ismcts60 | 3.5-6 | 104 | 10.4 | 35.0 | 0.023 | 0.344 |
| ismcts60 | 13-16 (peak) | 114 | 14.2 | 40.2 | 0.006 | 0.097 |
| ismcts60 | 28-31 | 128 | 25.9 | 41.1 | 0.019 | 0.028 |
| ismcts60 | 57-60 | 140 | 29.1 | 49.5 | 0.035 | 0.016 |

What changes after the peak (mix.png, games.png, losses.png):
- Passes per game double in ismcts60 (14 to 29), and claim_initiative per game goes from 7 to 12. Most of the growth in game length
  (114 to 140 decisions) is passes, then maneuvers and claim_initiative. Attacks per game fall.
  The play mix does not collapse: pass rises from 12% to 21% of plays, and every kind stays in use.
- The value net fits its training rows almost perfectly. Squared error against the game
  result falls to 0.016, near the 0.0074 floor of the 16-bin target, and the value loss is
  within 0.03-0.05 of its entropy floor. "Lv flat at 0.57" is this floor, not a plateau.
  A true opening-position predictor cannot reach this in a stochastic, hidden-information
  game with temp 0.5 and root noise. The net memorizes the result of each buffered game:
  about 130 rows per game share one z, and each row is trained 6 times during its ~500 s in the buffer.
  The peak in both runs comes when this error falls through about 0.08-0.1.
- After the peak, target entropy rises (1.75 to 1.96) and KL falls. This fits search
  Q values that no longer separate moves, so visits spread out and passive moves win more often.

Ruled out: the lr schedule (the drop starts 30 min before decay), seat bias (white share of
decided games stays at 0.50), buffer staleness (span flat
at ~500 s, cap full from 5 min), throughput changes (evals/s, rows/s and GPU busy stay flat),
and a collapse to one action kind. Timeouts rise a little (0.006 to 0.035) but stay small.

Main hypothesis: value overfitting to buffered game results. The self-play net then drifts
to passive play. The next steps are a holdout value loss on fresh games (the direct test), then
fewer optimizer passes per row or fewer rows per game, one change at a time.
