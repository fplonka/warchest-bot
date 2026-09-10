import bisect
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MATCHES = ROOT / "ladder/matches.jsonl"
BOTS = ROOT / "ladder/bots.json"
SCALE = math.log(10.0) / 400.0
GAMES, BLIND, GAUGE = 200, 10, 500.0


def load(index):
    played = []
    for line in MATCHES.read_text().splitlines():
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
            + 0.5 * (seat / 100.0) ** 2)


def contrasts_for(pairs, size):
    rows = torch.zeros(len(pairs), size + 2)
    for row, (a, b) in enumerate(pairs):
        rows[row, a], rows[row, b] = 1.0, -1.0
    return rows


def fit(played, size):
    contrasts = contrasts_for([(a, b) for a, b, _, _ in played], size)
    counts = torch.stack([c for _, _, c, _ in played])
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


def average(members, ages, size):
    """Contrasts for a branch's mean Elo at each age, and its mean improvement."""
    rows = {}
    for age in sorted({a for at in members for a in at}):
        column = torch.zeros(size + 2)
        for at in members:
            column[at[age]] = 1.0 / len(members)
        rows[age] = column
    gain = rows[max(rows)] - rows[min(rows)]
    return rows, gain


def report(names, ages, mean, covariance, played):
    runs = {}
    for i, name in enumerate(names):
        runs.setdefault(name.split(".")[0], {})[ages[i]] = i
    branches = {}
    for run, at in runs.items():
        branches.setdefault(run.rsplit("_s", 1)[0], []).append((run, at))
    size = len(names)

    lines = []
    for branch, members in branches.items():
        columns, gain = average([at for _, at in members], ages, size)
        elo = {age: float(column @ mean) for age, column in columns.items()}
        spread = float((gain @ covariance @ gain).sqrt())
        lines.append((float(gain @ mean), branch, elo, float(gain @ mean), spread, False))
        for run, at in sorted(members):
            early, late = at[min(at)], at[max(at)]
            lines.append((float(gain @ mean), "  " + run, {a: float(mean[at[a]]) for a in at},
                          float(mean[late] - mean[early]),
                          float((covariance[early, early] + covariance[late, late]
                                 - 2 * covariance[early, late]).sqrt()), True))
    lines.sort(key=lambda row: -row[0])

    ordered = {age: sorted(elo[age] for _, _, elo, _, _, dim in lines if not dim)
               for age in {a for _, _, elo, _, _, _ in lines for a in elo}}
    ordered["gain"] = sorted(g for _, _, _, g, _, dim in lines if not dim)

    print(f"{'branch':24s} {'30min':>8s} {'60min':>8s} {'120min':>8s} {'30->120':>15s}")
    for _, label, elo, gain, spread, dim in lines:
        ages = sorted(elo)
        cells = "".join(paint(f" {elo[a]:+8.0f}", elo[a],
                              None if a == ages[0] else ordered[a], dim)
                        for a in ages)
        tail = paint(f"{gain:+7.0f} +- {spread:<4.0f}", gain, ordered["gain"], dim)
        print(paint(f"{label:24s}", gain, ordered["gain"], dim) + cells + "   " + tail)
    fresh = sum(1 for *_, seed in played if seed != 83)
    print(f"\n{len(played)} matches ({fresh} on fresh openings), {len(names)} bots, "
          f"seat {mean[-2]:+.1f} Elo, draws {mean[-1].exp():.3f}")


def main():
    bots = json.loads(BOTS.read_text())
    names = sorted(bots)
    ages = [round(bots[name][0]) for name in names]
    played = load({name: i for i, name in enumerate(names)})
    mean, covariance = fit(played, len(names))
    if sys.argv[1:2] == ["report"]:
        report(names, ages, mean, covariance, played)
        return
    seed = 1000 + len(MATCHES.read_text().splitlines())
    pairs, gain = sharpening(mean, covariance, len(names))
    blind = seed % BLIND == 0
    pick = pairs[int(torch.randint(len(pairs), (1,), generator=torch.Generator()
                                   .manual_seed(seed)))] if blind else pairs[int(gain.argmax())]
    a, b = names[pick[0]], names[pick[1]]
    print(f"{bots[a][1]} {bots[b][1]} {GAMES} {seed} "
          f"{'blind' if blind else f'value({gain.max().sqrt():.1f})'} {a} {b}")


if __name__ == "__main__":
    main()
