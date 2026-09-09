import json
import math
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MATCHES = ROOT / "ladder/matches.jsonl"
NAME = re.compile(r"^([a-z0-9-]+)_s([12])\.(\S+)$")
AGES = (30.0, 60.0, 120.0)
SCALE = math.log(10.0) / 400.0
SCALES = (20.0, 11.0, 23.0, 10.5)
RUNG, TAILS, SIZE, GAMES, SURVEY, SAMPLES, QUADRATURE, AUDIT = 3.0, 4.0, 4, 400, 300, 512, 9, 5
SHAPE = 15.0
DISPERSION = 1.19


def parse(name, minutes):
    found = NAME.match(name)
    k = min(range(len(AGES)), key=lambda i: abs(AGES[i] - minutes))
    return found and abs(AGES[k] - minutes) < 5.0 and (found.group(1), name.split(".")[0], k)


def tally(outcomes):
    played = torch.tensor(outcomes)
    return torch.stack([(played.sign() == side).sum(0) for side in (1, -1, 0)], 1).float()


def load():
    matches = []
    for line in MATCHES.read_text().splitlines():
        row = json.loads(line)
        a, b = parse(row["a"], row["a_minutes"]), parse(row["b"], row["b_minutes"])
        if a and b and a[2] == b[2]:
            matches.append((a, b, tally(row["outcomes"]), row["draft_seed"]))
    return matches


class Model:
    """Strength relative to master: a_c + b_c x + e0[run] + e1[run] x."""

    def __init__(self, bots):
        self.branches = sorted({bot[0] for bot in bots} - {"master"})
        self.runs = sorted({bot[1] for bot in bots})
        self.n, self.r = len(self.branches), len(self.runs)
        self.size = 2 * self.n + 2 * self.r + 3

    def contrast(self, a, b):
        d = torch.zeros(self.size)
        for (branch, run, k), sign in ((a, 1.0), (b, -1.0)):
            x = float(k) - 1.0
            if branch != "master":
                c = self.branches.index(branch)
                d[c] += sign
                d[self.n + c] += sign * x
                d[-3] += sign * (x * x - 2.0 / 3.0)
            i = self.runs.index(run)
            d[2 * self.n + i] += sign
            d[2 * self.n + self.r + i] += sign * x
        return d

    def spans(self):
        n, r = self.n, self.r
        return ((0, n), (n, 2 * n), (2 * n, 2 * n + r), (2 * n + r, 2 * n + 2 * r))

    def blocks(self, theta):
        return [theta[start:end] for start, end in self.spans()]

    def load(self, rung):
        load = torch.zeros(self.n, self.size)
        load[range(self.n), range(self.n)] = 1.0
        load[range(self.n), [self.n + c for c in range(self.n)]] = rung
        return load


def logprobs(delta, log_nu):
    half = delta / 2.0
    total = torch.logaddexp(torch.logaddexp(half, -half), log_nu.expand_as(half))
    return torch.stack([half - total, -half - total, log_nu.expand_as(half) - total], -1)


def objective(theta, model, contrasts, counts, precisions):
    curve, seat, log_nu = theta[-3], theta[-2], theta[-1]
    delta = (contrasts @ theta) * SCALE
    both = torch.stack([delta + seat * SCALE, delta - seat * SCALE], 1)
    penalty = 0.5 * (precisions * theta[:-3] ** 2).sum()
    return (-(counts * logprobs(both, log_nu)).sum() / DISPERSION + penalty
            + 0.5 * (seat / 100.0) ** 2 + 0.5 * (curve / 100.0) ** 2)


def fit(model, matches, rounds=12):
    """Empirical Bayes: Laplace fit inside, EM update of the prior scales outside."""
    contrasts = torch.stack([model.contrast(a, b) for a, b, _, _ in matches])
    counts = torch.stack([c for _, _, c, _ in matches])
    scales = torch.tensor(SCALES)
    weights = torch.ones(model.size - 3)
    mean, covariance = torch.zeros(model.size), None
    for _ in range(rounds):
        precisions = weights / torch.cat([
            scales[i].expand(end - start) for i, (start, end) in enumerate(model.spans())]) ** 2
        theta = mean.clone().requires_grad_(True)
        optimiser = torch.optim.LBFGS([theta], max_iter=400, tolerance_grad=1e-10,
                                      line_search_fn="strong_wolfe")

        def closure():
            optimiser.zero_grad()
            loss = objective(theta, model, contrasts, counts, precisions)
            loss.backward()
            return loss

        optimiser.step(closure)
        mean = theta.detach()
        hessian = torch.autograd.functional.hessian(
            lambda t: objective(t, model, contrasts, counts, precisions), mean)
        covariance = torch.linalg.inv(hessian)
        energy = mean[:-3] ** 2 + covariance.diagonal()[:-3]
        scales = torch.stack([(weights[slice(*span)] * energy[slice(*span)]).sum().sqrt()
                              / weights[slice(*span)].sum().sqrt() for span in model.spans()])
        spread = torch.cat([scales[i].expand(end - start)
                            for i, (start, end) in enumerate(model.spans())]) ** 2
        weights = (TAILS + 1.0) / (TAILS + energy / spread)
    return mean, covariance, scales


def information(delta, nu):
    win, loss = torch.exp(delta / 2), torch.exp(-delta / 2)
    total = win + loss + nu
    return (win * (2 * loss + nu) ** 2 + loss * (2 * win + nu) ** 2
            + nu * (win - loss) ** 2) / (4 * total ** 3)


def precision(mean, contrasts, games):
    seat, nu = mean[-2] * SCALE, mean[-1].exp()
    delta = (contrasts @ mean) * SCALE
    return (games / 2) * SCALE ** 2 / DISPERSION * (information(delta + seat, nu)
                                                   + information(delta - seat, nu))


def regret(value, spread, noise, size):
    draws = value.unsqueeze(-2) + noise @ torch.linalg.cholesky(
        spread + 1e-6 * torch.eye(spread.shape[-1])).transpose(-1, -2)
    draws = draws.reshape(-1, *draws.shape[-2:])
    rows = torch.arange(draws.shape[0])
    best = torch.full(draws.shape[:2], -1e9)
    taken = torch.zeros(draws.shape[0], draws.shape[2], dtype=torch.bool)
    for _ in range(size):
        gain = torch.maximum(best.unsqueeze(2), draws).mean(1).masked_fill(taken, -1e9)
        pick = gain.argmax(1)
        taken[rows, pick] = True
        best = torch.maximum(best, draws[rows, :, pick])
    return (draws.max(2).values - best).mean(1)


def target(model, mean, covariance):
    """Strength at the rung. Extrapolating the line past the measured 120
    minutes carries a per-branch error no ladder game can reduce."""
    load = model.load(RUNG)
    drift = (SHAPE * (RUNG - 1.0)) ** 2 * torch.eye(model.n)
    return load @ mean, load @ covariance @ load.T + drift, load


def value_of(model, mean, covariance, pairs):
    """Expected reduction in shortlist regret for each candidate match."""
    size, games, survey, samples, quadrature = SIZE, GAMES, SURVEY, SAMPLES, QUADRATURE
    value, spread, load = target(model, mean, covariance)
    contrasts = torch.stack([model.contrast(a, b) for a, b in pairs])
    shift = contrasts @ covariance @ load.T
    denominator = ((contrasts @ covariance * contrasts).sum(1)
                   + 1.0 / precision(mean, contrasts, games))
    keep = torch.argsort((shift ** 2).sum(1) / denominator, descending=True)[:survey]
    shift = shift[keep] / denominator[keep].sqrt().unsqueeze(1)
    posterior = spread - shift.unsqueeze(2) * shift.unsqueeze(1)

    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(samples, model.n, generator=generator)
    nodes, weights = torch.linalg.eigh(
        torch.diag((torch.arange(1, quadrature, dtype=torch.float) / 2).sqrt(), 1)
        + torch.diag((torch.arange(1, quadrature, dtype=torch.float) / 2).sqrt(), -1))
    nodes, weights = nodes * math.sqrt(2.0), weights[0] ** 2
    means = value + shift.unsqueeze(1) * nodes.view(1, -1, 1)
    after = (regret(means, posterior.unsqueeze(1), noise, size)
             .view(len(keep), quadrature) * weights).sum(1)
    before = regret(value.unsqueeze(0), spread.unsqueeze(0), noise, size)[0]
    return keep, before - after


def pending(matches, bots):
    """The next match: one in AUDIT is drawn blind, the rest by decision value."""
    names = {bot: name for name, (minutes, _) in bots.items()
             if (bot := parse(name, minutes))}
    pairs = [(a, b) for i, a in enumerate(names) for b in list(names)[i + 1:]
             if a[2] == b[2]]
    order = torch.Generator().manual_seed(len(matches))
    if len(matches) % AUDIT == 0:
        a, b = pairs[int(torch.randint(len(pairs), (1,), generator=order))]
        return names[a], names[b], "audit", None
    model = Model(names)
    mean, covariance, _ = fit(model, matches)
    keep, gain = value_of(model, mean, covariance, pairs)
    a, b = pairs[keep[int(gain.argmax())]]
    return names[a], names[b], "value", float(gain.max())


def check(model, mean, covariance, matches):
    """Posterior-predictive diagnostics: calibration, draws, fresh openings."""
    contrasts = torch.stack([model.contrast(a, b) for a, b, _, _ in matches])
    counts = torch.stack([c for _, _, c, _ in matches])
    seat, nu = mean[-2] * SCALE, mean[-1].exp()
    delta = (contrasts @ mean) * SCALE
    both = torch.stack([delta + seat, delta - seat], 1)
    chance = logprobs(both, mean[-1]).exp()
    games = counts.sum((1, 2))
    expected = (chance[..., 0] + 0.5 * chance[..., 2]).mul(counts.sum(2)).sum(1)
    seen = (counts[..., 0] + 0.5 * counts[..., 2]).sum(1)
    spread = (games * (chance[..., 0] + 0.25 * chance[..., 2]
                       - (chance[..., 0] + 0.5 * chance[..., 2]) ** 2).mean(1)).sqrt()
    residual = (seen - expected) / spread
    fresh = torch.tensor([seed != 83 for *_, seed in matches])
    print(f"calibration  standardised residual sd {residual.std():.2f} "
          f"(1.00 = the stated uncertainty is right)")
    if fresh.sum() > 1:
        print(f"             seed 83 {residual[~fresh].std():.2f} over {(~fresh).sum()},"
              f" fresh openings {residual[fresh].std():.2f} over {fresh.sum()}")
    print(f"draws        predicted {chance[..., 2].mean():.3f}   "
          f"seen {(counts[..., 2].sum() / games.sum()):.3f}")
    residual_by_age = {}
    for (a, _, _, _), r in zip(matches, residual):
        residual_by_age.setdefault(AGES[a[2]], []).append(float(r))
    print("age linearity" + "".join(
        f"   {age:.0f}min {torch.tensor(v).mean():+.2f}"
        for age, v in sorted(residual_by_age.items())))


def main():
    bots = json.loads((ROOT / "ladder/bots.json").read_text())
    matches = load()
    if sys.argv[1:2] == ["report"]:
        model = Model([bot for name, (minutes, _) in bots.items()
                       if (bot := parse(name, minutes))])
        mean, covariance, scales = fit(model, matches)
        value, covar, _ = target(model, mean, covariance)
        spread = covar.diagonal().sqrt()
        print(f"{len(matches)} matches   scales {[round(float(x), 1) for x in scales]}   "
              f"seat {mean[-2]:+.1f}   draws {mean[-1].exp():.3f}")
        for i in torch.argsort(value, descending=True):
            print(f"  {model.branches[i]:16s} {value[i]:+7.1f} +- {spread[i]:.1f}")
        check(model, mean, covariance, matches)
        return
    a, b, kind, gain = pending(matches, bots)
    seed = 1000 + len(MATCHES.read_text().splitlines())
    print(f"{bots[a][1]} {bots[b][1]} {GAMES} {seed} "
          f"{kind}{'' if gain is None else f'({gain:.2f})'} {a} {b}")


if __name__ == "__main__":
    main()
