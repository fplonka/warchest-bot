# Two ideas to try next

Written 2026-11-10 after a review of the engine with the 295 h teacher net
(`runs/1028-teacher`, 1021-grow12-s99 snap 121). The numbers come from throwaway
probes on the Mac. They are small samples. Check them before you rely on them.

## What the probes showed

- The loop has reached a fixed point. On 109 self-play positions, judged by a
  16k-sim search, the 32-sim policy target is no better than the net's own prior
  (0.0000 +- 0.0003 value per decision). 64 sims gain 0.0004 and 1k sims gain 0.0013.
  So self-play search now teaches the policy almost nothing. Only the value head
  still learns, from one noisy result per game.
- The ladders agree. 1k -> 65k sims is +14 Elo. Training gives about 35 Elo per
  doubling from 8 h to 295 h.
- The search cannot see past the next draw. Each new hand makes a new node, so
  the tree stays inside the current round and the value net does the rest.
- The value net is good on average: against 96 playouts per move its values are
  calibrated (slope 0.99). 70 random moves that the search never visited were all
  worse than its pick (0.53 value worse on average).
- But moves that the search rates as equal are not equal. For 107 move pairs with
  a claimed gap near zero, playouts 6 to 12 decisions deep show a true gap of about
  0.02 value RMS (the interval excludes zero). A direct test that overrules the
  search with playouts gave +0.004 +- 0.002 per decision on 81 positions. That is
  a hint, not proof.
- Random drafts hide skill. The median draft has a pre-game favourite of about
  78%, and 42% of self-play decisions are in positions with |value| > 0.9. In
  `ladder/td2w`, a bot that scores 58-70% overall is +123 Elo on balanced drafts
  and +44 on the most lopsided 28%.

Tried and flat: 20 whole-game biases on the raw policy (recruit, attack, pass and
so on, +-1 logit, 600 paired drafts each) all scored within about 1.5 points of
50%. Luck-adjusted results cut variance only 1.4x. The same random draws for all
root moves inside the tree search changed nothing measurable.

## Idea 1: compare moves on the same future luck

The problem: the tree search stops at the draw, so it cannot separate moves whose
difference shows two or three rounds later. The value net cannot separate them
either, because it only learns from win or loss.

The idea: for the few best moves at the root, play each one forward with the raw
policy for both sides, about four rounds, many times. Give every move the same
opponent hand and the same future draws. The luck is then the same for all
moves, so it cancels, and the small real difference is left. Read the value net
at the end of each playout and average.

Use it in two places:

1. At play time. Extra tree sims are worth nothing above about 1k. Spend that
   budget on these playouts for the top moves when the search says they are close
   and the game is not yet decided. Rough estimate: +50 to +70 Elo. Test: plain
   search at 1k sims against the same search plus playouts, on balanced drafts.
2. In training. Use the playout comparison as the policy target at close
   decisions, in place of a target that now carries no information. A second form
   is to fork a self-play game at a close decision, play both moves to the end on
   the same draws, and train on the difference in result. If the play-time test
   wins, this is the larger prize, because each generation then sees further than
   one round and the gain compounds.

What can go wrong: the playouts use the raw policy, not the search, so they
answer "which move is better if both sides then play fast". If that differs from
the answer under full search, the gain is smaller than the estimate.

## Idea 2: train an adversary

The problem: every probe above tests one move at a time on positions the bot
reaches by itself. A plan that needs several rounds of moves the bot never plays
is invisible to all of them, and to self-play. The bot only knows the results of
games it has played.

The idea: freeze the teacher. Train a copy whose only goal is to beat it. The
copy plays only against the frozen bot, with more random move choice than
self-play uses, and learns from its own results.

Read the result as follows:

- The copy stays near 50-55%: the teacher has no large hole. The plateau is the
  game, and strength work can stop.
- The copy wins clearly: look at how. Then add it (and later ones) as opponents
  in the main run, so the teacher must fix the hole.

This is the one experiment that can show a large gain exists. A day on the box
is enough to know.

## Do first: a ladder on balanced drafts

Deal only drafts with pre-game |value| < 0.5 by the current best net. The same
skill gap is then about 1.4x larger in Elo, and a match needs about 1.8x fewer
games. Both ideas above are tests of 50 Elo or less on the old ladder, so this
makes them cheaper to judge.
