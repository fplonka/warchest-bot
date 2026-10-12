
import argparse
import collections
import copy
import dataclasses
import gzip
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from numpy.lib.format import read_array, write_array

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn as nn
import torch.nn.functional as F

import warchest
import config
import export_weights
from value_net import Net

ROOT = pathlib.Path(__file__).resolve().parent.parent

ROW_BYTES = warchest.ROW_BYTES
ACT_BYTES = warchest.ACT_BYTES
EMA_ROWS = 1_024_000
HORIZONS = warchest.HORIZONS
SHORTS = tuple(f"short_{h:g}_loss" for h in HORIZONS[1:])
SHORT_W = 0.1
WINDOW = 250_000
OWN_W = 1.0
UNSEEN_W = 0.1
PLAYS = ("attack", "pass", "deploy", "bolster", "maneuver", "recruit", "claim_initiative")
FRESH = ("value_loss", "value_sq_err", "policy_loss", "own_loss", "unseen_loss", *SHORTS)
GAME_KEYS = ("games", "decisions", "timeouts", "white_wins", "black_wins", "draws",
             *(f"plays_{p}" for p in PLAYS))


def adam(net, lr, wd):
    wide = [p for p in net.parameters() if p.dim() >= 2]
    thin = [p for p in net.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": wide, "weight_decay": wd}, {"params": thin, "weight_decay": 0.0}],
        lr=lr, fused=True)


def tail(a, n):
    return a[len(a) - n:]


class Buffer:

    def __init__(self, cap):
        self.cap, self.acap = cap, cap * 32
        self.x = np.zeros((cap, ROW_BYTES), np.uint8)
        self.z = np.zeros((cap, len(HORIZONS) + warchest.N_LOCATIONS + warchest.NSLOT), np.float32)
        self.astart = np.zeros(cap, np.int64)
        self.alen = np.zeros(cap, np.int32)
        self.written_at = np.zeros(cap, np.float64)
        self.pa = np.zeros((self.acap, ACT_BYTES), np.uint8)
        self.pp = np.zeros(self.acap, np.float16)
        self.rows = self.acts = self.lo = 0

    def add(self, rows, desc, off, target, z):
        n, na = len(rows), len(desc)
        while self.lo < self.rows and (self.rows - self.lo + n > self.window(self.rows + n)
                                       or self.acts - self.astart[self.lo % self.cap] + na > self.acap):
            self.lo += 1
        ring = (np.arange(n) + self.rows) % self.cap
        self.x[ring], self.z[ring] = rows, z
        self.astart[ring] = self.acts + off[:-1]
        self.alen[ring] = np.diff(off)
        self.written_at[ring] = time.time()
        at = (np.arange(na) + self.acts) % self.acap
        self.pa[at], self.pp[at] = desc, target
        self.rows += n
        self.acts += na

    def window(self, seen):
        return min(self.cap, int(WINDOW * (1 + 0.4 * ((max(seen, WINDOW) / WINDOW) ** 0.75 - 1) / 0.75)))

    def live_rows(self):
        return np.arange(self.lo, self.rows) % self.cap

    def live_acts(self):
        return np.arange(self.astart[self.lo % self.cap], self.acts) % self.acap

    def save(self, path):
        rows, acts = self.live_rows(), self.live_acts()
        with gzip.open(path, "wb", compresslevel=1) as f:
            write_array(f, np.array([self.lo, self.rows, self.acts]))
            write_array(f, time.time() - self.written_at[rows])
            for a in (self.x, self.z, self.astart, self.alen):
                write_array(f, a[rows])
            for a in (self.pa, self.pp):
                write_array(f, a[acts])

    def load(self, path):
        with gzip.open(path) as f:
            lo, self.rows, self.acts = read_array(f).tolist()
            self.lo = max(lo, self.rows - self.cap)
            rows = self.live_rows()
            self.written_at[rows] = time.time() - tail(read_array(f), len(rows))
            self.x[rows] = tail(read_array(f), len(rows))
            z = tail(read_array(f), len(rows))
            self.z[rows, :z.shape[1]] = z
            for a in (self.astart, self.alen):
                a[rows] = tail(read_array(f), len(rows))
            self.lo += int(np.searchsorted(self.astart[rows], self.acts - self.acap))
            acts = self.live_acts()
            for a in (self.pa, self.pp):
                a[acts] = tail(read_array(f), len(acts))
        return self

    def span_seconds(self):
        return time.time() - self.written_at[self.lo % self.cap] if self.lo < self.rows else 0.0

    def __len__(self):
        return self.rows - self.lo

    def gather(self, ids):
        s = ids % self.cap
        lens = self.alen[s].astype(np.int64)
        within = np.arange(int(lens.sum())) - np.repeat(np.cumsum(lens) - lens, lens)
        at = (np.repeat(self.astart[s], lens) + within) % self.acap
        parow = np.repeat(np.arange(len(ids)), lens)
        return self.x[s], self.pa[at], parow, self.pp[at].astype(np.float32), self.z[s]


@torch.compile(dynamic=True)
def losses(net, x, desc, parow, target, z, wp):
    with torch.autocast("cuda", torch.bfloat16):
        win, short, pieces = net.evaluate(x)
        logit = net.actions(desc, pieces, parow)
        own, unseen = net.aux_heads(pieces)
    win, short, logit, own, unseen = win.float(), short.float(), logit.float(), own.float(), unseen.float()
    held, coins = z[:, len(HORIZONS):].split([warchest.N_LOCATIONS, warchest.NSLOT], 1)
    value_loss = F.binary_cross_entropy_with_logits(win, (z[:, 0] + 1) / 2)
    short_loss = (short - z[:, 1:len(HORIZONS)]).square().mean(0)
    own_loss = F.binary_cross_entropy_with_logits(own, (held + 1) / 2)
    unseen_loss = (unseen - coins).square().mean()
    n = x.shape[0]
    top = torch.full((n,), -1e30, device=x.device).scatter_reduce(0, parow, logit.detach(), reduce="amax")
    shifted = logit - top[parow]
    total = torch.zeros(n, device=x.device).index_add_(0, parow, shifted.exp())
    logp = shifted - total[parow].log()
    taught = target.sum()
    policy_loss = -(target * logp).sum() / taught
    entropy = -(target * target.clamp(min=1e-30).log()).sum() / taught
    stats = {"value_loss": value_loss.detach(),
             "policy_loss": policy_loss.detach(),
             "own_loss": own_loss.detach(),
             "unseen_loss": unseen_loss.detach(),
             "policy_target_entropy": entropy,
             "policy_kl": policy_loss.detach() - entropy,
             "value_sq_err": (net.value(win.detach()) - z[:, 0]).square().mean(),
             **dict(zip(SHORTS, short_loss.detach().unbind()))}
    return value_loss + wp * policy_loss + SHORT_W * short_loss.sum() + OWN_W * own_loss + UNSEEN_W * unseen_loss, stats


def train_steps(net, acting, opt, buf, steps, batch, micro, rng, device, batch_fn, policy_w):
    acc = collections.Counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(batch // micro):
            parts = batch_fn(buf.gather(rng.integers(buf.lo, buf.rows, size=micro)), device)
            loss, stats = losses(net, *parts, policy_w)
            (loss * (micro / batch)).backward()
            for key, value in stats.items():
                acc[key] = acc[key] + value * (micro / batch)
        acc["grad_norm"] = acc["grad_norm"] + nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(acting.parameters()), list(net.parameters()), batch / EMA_ROWS)
    if not steps:
        return 0, {}
    return steps, dict(zip(acc, (torch.stack([v.float() for v in acc.values()]) / steps).tolist()))


def probe(net, buf, n, micro, rng, device, batch_fn, policy_w):
    parts = batch_fn(buf.gather(rng.integers(buf.rows - n, buf.rows, size=min(n, micro))), device)
    with torch.no_grad():
        _, stats = losses(net, *parts, policy_w)
    return {f"fresh_{k}": stats[k] * n for k in FRESH}


def ingest(buf, data):
    rows = np.asarray(data["rows"], np.uint8).reshape(-1, ROW_BYTES)
    if len(rows):
        buf.add(rows, np.asarray(data["desc"], np.uint8).reshape(-1, ACT_BYTES),
                np.asarray(data["off"], np.int64), np.asarray(data["target"], np.float16),
                data["z"])
    return len(rows)


def append_epoch(args, rec):
    with open(f"{args.out}/epochs.jsonl", "a") as f:
        f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def rewind_epochs(args, epoch):
    path = f"{args.out}/epochs.jsonl"
    with open(path) as f:
        kept = [line for line in f if json.loads(line)["epoch"] < epoch]
    with open(path, "w") as f:
        f.writelines(kept)


def write_log(args, snaps):
    path = f"{args.out}/log.json"
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"cfg": dataclasses.asdict(args), "snapshots": snaps}, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def buffer_file(snapshot):
    return pathlib.Path(snapshot).with_suffix(".buffer")


def cpu_state(net):
    return {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}


def publish_state(state):
    net = Net()
    net.load_state_dict(state)
    net.push()


def main():
    ap = argparse.ArgumentParser(description="Train one run, packing every snapshot as a bot.")
    ap.add_argument("over", nargs="*", help="knob=value (production defaults)")
    over = config.parse(ap.parse_args().over)
    resume = over.pop("resume", "")
    checkpoint = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    base = config.Cfg(**checkpoint["cfg"]) if checkpoint else config.BASELINE
    args = dataclasses.replace(base, **over)
    if not args.out:
        raise SystemExit("pass out=<name>")
    if not args.out.startswith("runs/"):
        args.out = f"runs/{args.out}"
    args.git = config.git_sha()
    if resume:
        if not os.path.isdir(args.out):
            raise SystemExit(f"resume output {args.out} does not exist")
    else:
        if os.path.exists(args.out):
            raise SystemExit(f"{args.out} exists")
        os.makedirs(args.out)
    logf = open(f"{args.out}/train.log", "a" if resume else "w")

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            logf.write(s)
            return len(s)

        def flush(self):
            sys.__stdout__.flush()
            logf.flush()
    sys.stdout = sys.stderr = Tee()
    print(f"[resume] {resume}" if resume else
          f"[train] {args.out} at {args.git} seed={args.seed} {over or 'baseline'}", flush=True)
    if args.note:
        print(f"[train] {args.note}", flush=True)
    for bad, why in (
            (args.replay_ratio <= 0.0, "replay_ratio must be positive"),
            (args.target_every <= 0.0, "target_every must be positive minutes"),
            (args.snapshot_every <= 0.0, "snapshot_every must be positive minutes"),
            (not 0.0 <= args.warm_minutes <= args.minutes, "warm_minutes must be within the run"),
            (args.lr_final <= 0.0 or args.lr_final > args.lr, "lr_final must be in (0, lr]"),
            (args.lr_warmup <= 0.0, "lr_warmup must be positive minutes"),
            (not 0.0 <= args.lr_stable_frac < 1.0, "lr_stable_frac must be in [0, 1)"),
            (not 0.0 < args.pcr_full <= 1.0, "pcr_full must be in (0, 1]"),
            (args.batch % args.micro != 0, "micro must divide batch"),
            (not torch.cuda.is_available(), "training requires a working GPU")):
        if bad:
            raise SystemExit(why)

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    torch.cuda.set_stream(torch.cuda.Stream(dev, priority=torch.cuda.Stream.priority_range()[1]))

    import gpu_batch
    torch.manual_seed(args.seed)
    value = Net().to(dev)
    acting = copy.deepcopy(value).requires_grad_(False)
    opt = adam(value, args.lr, args.wd)
    buf = Buffer(args.cap)
    origin = resume or args.init_weights
    if origin and buffer_file(origin).exists():
        buf.load(buffer_file(origin))
    if checkpoint:
        value.load_state_dict(checkpoint["value"])
        acting.load_state_dict(checkpoint["acting"])
        opt.load_state_dict(checkpoint["optimizer"])
        target_state = checkpoint["target"]
        publish_state(target_state)
        rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    else:
        if args.init_weights:
            grown = export_weights.load(args.init_weights).state_dict()
            for net in (value, acting):
                net.load_state_dict(grown)
        acting.push()
        target_state = cpu_state(acting)

    total, warm = args.minutes * 60.0, args.warm_minutes * 60.0
    snap_gap = args.snapshot_every * 60.0
    elapsed = float(checkpoint["elapsed"]) if checkpoint else 0.0
    next_snap = float(checkpoint["next_snapshot"]) if checkpoint else snap_gap
    t0 = time.time() - elapsed
    epoch = int(checkpoint["epoch"]) if checkpoint else 0
    if checkpoint:
        rewind_epochs(args, epoch)
    progress = checkpoint["progress"] if checkpoint else {
        "optimizer_rows": 0, "generated_rows": 0, "farm_runs": 0, "search_decisions": 0, "search_rows": 0, "kinds": {}}
    snaps = [s for s in checkpoint["snapshots"] if os.path.exists(f"{args.out}/{s['file']}")] if checkpoint else []
    write_log(args, snaps)

    def snapshot(el, label=None):
        n = int(snaps[-1]["file"][5:9]) + 1 if snaps else 0
        label = label or f"s{n}"
        path = f"{args.out}/snap_{n:04d}.pt"
        snaps.append({"label": label, "t": round(el, 1), "file": os.path.basename(path)})
        state = {
            "value": value.state_dict(), "acting": acting.state_dict(), "optimizer": opt.state_dict(),
            "target": target_state, "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "elapsed": float(el), "next_snapshot": float(el + snap_gap),
            "epoch": epoch, "snapshots": snaps, "progress": progress,
            "cfg": dataclasses.asdict(args), "label": label, "git": args.git,
        }
        replay = buffer_file(path)
        buf.save(f"{replay}.tmp")
        os.replace(f"{replay}.tmp", replay)
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)
        for old in snaps[:-1]:
            buffer_file(f"{args.out}/{old['file']}").unlink(missing_ok=True)
            os.remove(f"{args.out}/{old['file']}")
        del snaps[:-1]
        write_log(args, snaps)
        export_weights.bot(args.out, label, acting, args.git, el / 60.0)
        print(f"[t={el:6.1f}s] --- snapshot {snaps[-1]['file']} ({label}) ---", flush=True)

    def tick(rec, line):
        nonlocal epoch, next_snap
        rec.update(t=round(time.time() - t0, 1), epoch=epoch, buf=len(buf), lr=opt.param_groups[0]["lr"])
        append_epoch(args, rec)
        print(line, flush=True)
        epoch += 1
        if time.time() - t0 >= next_snap:
            snapshot(time.time() - t0)
            next_snap = time.time() - t0 + snap_gap

    def fit():
        for group in opt.param_groups:
            group["lr"] = args.lr_at(time.time() - t0)
        debt = args.replay_ratio * progress["generated_rows"] - progress["optimizer_rows"]
        steps = int(debt // args.batch) if len(buf) >= args.batch else 0
        done, stats = train_steps(value, acting, opt, buf, steps, args.batch, args.micro, rng, dev, gpu_batch.make_batch,
                                  args.policy_w)
        progress["optimizer_rows"] += done * args.batch
        return done, stats

    def report(phase, window, stats, extra):
        games = max(window["games"], 1)
        rec = {"phase": phase, **{k: int(window[k]) for k in GAME_KEYS},
               "rows": int(window["rows"]), "policy_rows": int(window["policy_rows"]), "steps": int(window["steps"]),
               "optimizer_rows": progress["optimizer_rows"], "replay_rows": progress["generated_rows"],
               "timeout_frac": round(window["timeouts"] / games, 3), "buf_s": round(buf.span_seconds(), 1),
               "tgt_mean": round(window["z_sum"] / max(window["rows"], 1), 4),
               **{f"fresh_{k}": round(float(window[f"fresh_{k}"]) / max(window["rows"], 1), 5) for k in FRESH},
               "kinds": {k: v for k, v in progress["kinds"].items() if v[1]},
               **{k: round(v, 5) for k, v in stats.items()}, **extra}
        tick(rec, f"[t={time.time() - t0:6.1f}s] {phase} games={rec['games']} rows={rec['rows']} "
                  f"W{rec['white_wins']}/B{rec['black_wins']}/D{rec['draws']}/T{rec['timeouts']} "
                  f"steps={rec['steps']} Lv={stats.get('value_loss', float('nan')):.4f} "
                  f"Lp={stats.get('policy_loss', float('nan')):.4f} "
                  + " ".join(f"{k}={v}" for k, v in extra.items()))

    def absorb(window, data):
        n = ingest(buf, data)
        progress["generated_rows"] += n
        window["rows"] += n
        window["policy_rows"] += int(np.count_nonzero(np.diff(data["off"])))
        window["z_sum"] += float(data["z"][:, 0].sum())
        for k in GAME_KEYS:
            window[k] += int(data[k])
        for kind, got in data["kinds"].items():
            progress["kinds"][kind] = [a + b for a, b in zip(progress["kinds"].get(kind, (0, 0)), got)]
        window.update(probe(value, buf, n, args.micro, rng, dev, gpu_batch.make_batch, args.policy_w))

    if not checkpoint:
        snapshot(0.0, "init")
    window = collections.Counter()
    while time.time() - t0 < warm:
        absorb(window, warchest.warm(args.warm_games, args.seed * 1_000_003 + epoch))
        done, stats = fit()
        window["steps"] += done
        report("warm", window, stats, {})
        window.clear()
    while time.time() - t0 < total:
        acting.push()
        target_state = cpu_state(acting)
        playout = args.playout_at(time.time() - t0)
        farm = warchest.Farm(args.seed + 1_000_003 * progress["farm_runs"], args.games,
                             list(range(torch.cuda.device_count())), *playout, chunk=args.collect_rows)
        progress["farm_runs"] += 1
        print(f"[t={time.time() - t0:6.1f}s] --- self-play: sims, fast_sims, p_full = {playout} ---", flush=True)
        next_target = time.time() + args.target_every * 60.0
        next_report = time.time() + 10.0
        at, seen = time.time(), (0, 0)
        stats = {}
        while time.time() - t0 < total and args.playout_at(time.time() - t0) == playout:
            absorb(window, farm.collect())
            done, fitted = fit()
            window["steps"] += done
            stats = fitted or stats
            if time.time() >= next_target:
                acting.push()
                target_state = cpu_state(acting)
                next_target += args.target_every * 60.0
                print(f"[t={time.time() - t0:6.1f}s] --- target network refresh ---", flush=True)
            if time.time() < next_report:
                continue
            now, span = time.time(), max(time.time() - at, 1e-9)
            calls, evals = map(sum, zip(*farm.gpu_stats()))
            progress["search_decisions"] += window["decisions"]
            progress["search_rows"] += window["rows"]
            searched = now - t0 - warm
            extra = {"evals_per_s": round((evals - seen[1]) / span, 1),
                     "decisions_per_s": round(progress["search_decisions"] / searched, 1),
                     "rows_per_s": round(progress["search_rows"] / searched, 1),
                     "rows_per_call": round((evals - seen[1]) / max(calls - seen[0], 1), 1)}
            seen = (calls, evals)
            report("search", window, stats, extra)
            window.clear()
            at, next_report = now, now + 10.0
        del farm
    snapshot(time.time() - t0, "final")


if __name__ == "__main__":
    main()
