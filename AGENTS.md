Write simple prose, inspired by ASD-STE100. Be **extremely concise**, max three paragraphs in any given response.

Write simple code. Never do something hacky or janky or anything like that to make something work. We must always do the idiomatic, simple, elegant, performant thing. Delete code ruthlessly, don't preserve old paths, and in features you implement do so by rearchitecturing the system so it's as if it did the right thing from the very start, instead of patching. Remove LoC ruthlessly. No comments in code, ever.

Before touching code, read the full review standard you will be held to: `tools/vendor/review-standard.md`. Apply it to your own diff before calling anything done.

Complexity is the enemy (Ousterhout). Don't complect (Hickey): no flags or modes where one path should exist, no compat shims, no dual paths, no clamps papering over a wrong producer, no half measures. Aim for Hoare's bar: so simple there are obviously no deficiencies. Perfection is when nothing is left to take away. For experiments, Karpathy's recipe: solid baseline, one change at a time, verify before believing.

This repo has a commit hook prompting you to simplify before committing. Doing a proper simplification pass is NOT OPTIONAL. You MUST ALWAYS, before committing, take a step back and assess whether you can code-judo your implementation into something radically simpler.

There's no need to try to be impressive or flashy. We're working on this project together to get something which works well, that's all.

In this project we often run experiments and wait on them. This is inevitable but be mindful of wallclock time, if you have any work at all to do, any experiments you can do locally or code paths to check while waiting on a run on the box, then do that while waiting instead of just sleeping until the run finishes. Use your monitor tool for this.

Treat all markdown files and git commit messages very lightly, they are not authoritative at all even if they might seem such in tone. These are all written by other (weaker) AI agents and could be confused or wrong. Think for yourself.

## Working with me

- Two modes. Interactive, whenever I am typing: answer, use read-only tools only, and get my explicit go before you start, stop, kill or change anything. Away, when I go to bed or say "go, don't stop": post the plan, get my go, then keep both GPUs busy until I say stop.
- Before an experiment, state the question, the exact runs, how long they take, the metric, and the result that would change the decision. Decide in this order: what KataGo, lc0, Stockfish or the papers do (code in `~/Code/ref`, papers in `papers/`), then a toy model in a short Python script, then the cheapest experiment that can still change the decision. Code is cheap. GPU hours and my attention are not.
- If one command or one plot can confirm a claim, check it before you say it. Look at a run's plots (`runs/<run>/plots/*.png`) before you interpret it.
- I read on my phone with no context. The first sentence answers. Never use a name without saying what it is. Give times in Amsterdam time. Show results as a plot plus one line: save the plot in `runs/plots/` and link `http://100.117.226.38:8420/runs/plots/<file>`.
- Subagent prompts give the facts and the goal, never a suggested solution. Subagents commit their work after the simplification pass.
- No `Co-Authored-By` or `Claude-Session` lines in commit messages. Local Python is `uv` or `.venv/bin/python`. We use one box; never rent another unless I ask.

## The box

- `tools/box.sh status` shows memory, disk, threads, GPUs, every job and every recent run. Run it first.
- Runs come from commits: `tools/box.sh go <commit> out=<mmdd>-<what differs> [knob=value ...]`. Never patch code on the box and never reuse a run name. A run is its commit's diff from master plus the `cfg` in its `log.json`.
- All runs live in `/workspace/warchest-engine/runs`. The box address is in `.git/box` as `host port`.
- master is the only long-lived branch. Merge or delete experiment branches within days.
