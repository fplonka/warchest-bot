import bisect
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCALE = math.log(10.0) / 400.0
GAMES, BLIND, GAUGE = 100, 10, 500.0


def load(path, index):
    played = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["a"] in index and row["b"] in index:
            counts = torch.zeros(2, 3)
            for pair in row["outcomes"]:
                for seat, value in enumerate(pair):
                    counts[seat][0 if value > 0 else 1 if value < 0 else 2] += 1
            played.append((index[row["a"]], index[row["b"]], counts, row["draft_seed"]))
    return played


def logprobs(delta, log_nu):
    half = delta / 2.0
    total = torch.logaddexp(torch.logaddexp(half, -half), log_nu.expand_as(half))
    return torch.stack([half - total, -half - total, log_nu.expand_as(half) - total], -1)


def objective(theta, contrasts, counts, size):
    seat, log_nu = theta[-2], theta[-1]
    delta = (contrasts @ theta) * SCALE
    both = torch.stack([delta + seat * SCALE, delta - seat * SCALE], 1)
    return (-(counts * logprobs(both, log_nu)).sum()
            + 0.5 * ((theta[:size] / GAUGE) ** 2).sum()
            + 0.5 * (seat / 100.0) ** 2
            + 0.5 * (log_nu / 3.0) ** 2)


def contrasts_for(pairs, size):
    rows = torch.zeros(len(pairs), size + 2)
    for row, (a, b) in enumerate(pairs):
        rows[row, a], rows[row, b] = 1.0, -1.0
    return rows


def fit(played, size):
    contrasts = contrasts_for([(a, b) for a, b, _, _ in played], size)
    counts = (torch.stack([c for _, _, c, _ in played]) if played
              else torch.zeros(0, 2, 3))
    theta = torch.zeros(size + 2, requires_grad=True)
    optimiser = torch.optim.LBFGS([theta], max_iter=600, tolerance_grad=1e-10,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimiser.zero_grad()
        loss = objective(theta, contrasts, counts, size)
        loss.backward()
        return loss

    optimiser.step(closure)
    mean = theta.detach()
    hessian = torch.autograd.functional.hessian(
        lambda t: objective(t, contrasts, counts, size), mean)
    return mean, torch.linalg.inv(hessian)


def information(delta, nu):
    win, loss = torch.exp(delta / 2), torch.exp(-delta / 2)
    total = win + loss + nu
    return GAMES * (win * (2 * loss + nu) ** 2 + loss * (2 * win + nu) ** 2
                    + nu * (win - loss) ** 2) / (4 * total ** 3) * SCALE ** 2


def sharpening(mean, covariance, size):
    pairs = [(a, b) for a in range(size) for b in range(a + 1, size)]
    contrasts = contrasts_for(pairs, size)
    seat, nu = mean[-2] * SCALE, mean[-1].exp()
    delta = (contrasts @ mean) * SCALE
    precision = 0.5 * (information(delta + seat, nu) + information(delta - seat, nu))
    shift = contrasts @ covariance
    return pairs, (shift * shift).sum(1) / ((shift * contrasts).sum(1) + 1.0 / precision)


RAMP = (196, 202, 208, 214, 220, 226, 190, 154, 118, 82, 46)


def paint(text, value, ordered, dim):
    if not sys.stdout.isatty():
        return text
    if dim or ordered is None:
        return f"\033[38;5;244m{text}\033[0m"
    step = bisect.bisect_left(ordered, value) / max(len(ordered) - 1, 1)
    return f"\033[38;5;{RAMP[round(step * (len(RAMP) - 1))]}m{text}\033[0m"


def optimizer_rows(entries):
    logs = {}
    for age, path in entries:
        log = ROOT / Path(path).parents[1] / "epochs.jsonl"
        if log not in logs:
            logs[log] = [(row["t"], row["optimizer_rows"])
                         for row in map(json.loads, log.read_text().splitlines())
                         if "optimizer_rows" in row]
        yield min(logs[log], key=lambda row: abs(row[0] - age * 60))[1]


def age_label(minutes):
    return f"{minutes / 60:g}h" if minutes >= 60 else f"{minutes:g}m"


def report(names, ages, trained, mean, covariance, played):
    runs = {}
    for i, name in enumerate(names):
        runs.setdefault(name.split(".")[0], {})[ages[i]] = i
    branches = {}
    for run, at in runs.items():
        branches.setdefault(run.rsplit("_s", 1)[0], []).append((run, at))

    def describe(members):
        columns = {}
        for age in sorted({a for at in members for a in at}):
            column = torch.zeros(len(names) + 2)
            present = [at[age] for at in members if age in at]
            column[present] = 1.0 / len(present)
            columns[age] = column
        gain = columns[max(columns)] - columns[min(columns)]
        value = float(gain @ mean)
        doublings = sum(math.log2(trained[at[max(at)]] / trained[at[min(at)]])
                        for at in members) / len(members)
        return ({age: float(c @ mean) for age, c in columns.items()}, value,
                float((gain @ covariance @ gain).sqrt()), value / doublings)

    blocks = []
    for branch, members in branches.items():
        head = describe([at for _, at in members])
        blocks.append((head[1], [(branch, *head, False)]
                       + [("  " + run, *describe([at]), True)
                          for run, at in sorted(members)]))
    lines = [row for _, block in sorted(blocks, key=lambda b: -b[0]) for row in block]

    bright = [row for row in lines if not row[-1]]
    every = sorted({a for _, elo, *_ in lines for a in elo})
    ordered = {age: sorted(elo[age] for _, elo, *_ in bright if age in elo)
               for age in every}
    ordered["gain"] = sorted(row[2] for row in bright)
    ordered["rows"] = sorted(row[4] for row in bright)

    span = f"{age_label(every[0])}->{age_label(every[-1])}"
    print(f"{'branch':24s}" + "".join(f" {age_label(a):>8s}" for a in every)
          + f" {span:>15s} {'Elo/2x rows':>12s}")
    for label, elo, gain, spread, per_row, dim in lines:
        cells = "".join(paint(f" {elo[a]:+8.0f}", elo[a], ordered[a], dim)
                        if a in elo else " " * 9 for a in every)
        print(paint(f"{label:24s}", gain, ordered["gain"], dim) + cells + "   "
              + paint(f"{gain:+7.0f} +- {spread:<4.0f}", gain, ordered["gain"], dim)
              + paint(f"{per_row:+12.0f}", per_row, ordered["rows"], dim))
    fresh = sum(1 for *_, seed in played if seed != 83)
    print(f"\n{len(played)} matches ({fresh} on fresh openings), {len(names)} bots, "
          f"seat {mean[-2]:+.1f} Elo, draws {mean[-1].exp():.3f}")


def main():
    campaign = Path(sys.argv[1])
    matches_path = campaign / "matches.jsonl"
    matches_path.touch()
    bots = {}
    for run in (campaign / "runs").read_text().split():
        for spec in sorted((ROOT / "runs" / run / "bots").glob("*/bot.json")):
            bot = json.loads(spec.read_text())
            if round(bot["minutes"] / 60) in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
                bots[bot["name"]] = (bot["minutes"], str(spec.parent.relative_to(ROOT)))
    names = sorted(bots)
    ages = [round(bots[name][0]) for name in names]
    played = load(matches_path, {name: i for i, name in enumerate(names)})
    mean, covariance = fit(played, len(names))
    if sys.argv[2:] == ["report"]:
        trained = list(optimizer_rows(bots[name] for name in names))
        report(names, ages, trained, mean, covariance, played)
        return
    seed = 1000 + len(matches_path.read_text().splitlines())
    pairs, gain = sharpening(mean, covariance, len(names))
    blind = seed % BLIND == 0
    pick = pairs[int(torch.randint(len(pairs), (1,), generator=torch.Generator()
                                   .manual_seed(seed)))] if blind else pairs[int(gain.argmax())]
    a, b = names[pick[0]], names[pick[1]]
    print(f"{bots[a][1]} {bots[b][1]} {GAMES} {seed} "
          f"{'blind' if blind else f'value({gain.max().sqrt():.1f})'} {a} {b}")


if __name__ == "__main__":
    main()
