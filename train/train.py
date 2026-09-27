
import argparse
import collections
import copy
import dataclasses
import json
import math
import os
import pathlib
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn as nn

import warchest
import config
import export_weights
from value_net import BINS, Net

ROOT = pathlib.Path(__file__).resolve().parent.parent

ROW_BYTES = warchest.ROW_BYTES
ACT_BYTES = warchest.ACT_BYTES
EMA = 0.999
PLAYS = ("attack", "pass", "deploy", "bolster", "maneuver", "recruit", "claim_initiative")
GAME_KEYS = ("games", "decisions", "timeouts", "white_wins", "black_wins", "draws",
             *(f"plays_{p}" for p in PLAYS))


def adam(net, lr, wd):
    wide = [p for p in net.parameters() if p.dim() >= 2]
    thin = [p for p in net.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": wide, "weight_decay": wd}, {"params": thin, "weight_decay": 0.0}],
        lr=lr, fused=True)


def scheduled_lr(initial, final, elapsed, duration, stable_frac):
    stable = stable_frac * duration
    if stable_frac >= 1.0 or elapsed <= stable:
        return initial
    if elapsed >= duration:
        return final
    return final + (initial - final) * (1.0 + math.cos(
        math.pi * ((elapsed - stable) / (duration - stable)))) / 2.0


class Buffer:

    def __init__(self, cap):
        self.cap, self.acap = cap, cap * 32
        self.x = np.zeros((cap, ROW_BYTES), np.uint8)
        self.z = np.zeros(cap, np.float32)
        self.astart = np.zeros(cap, np.int64)
        self.alen = np.zeros(cap, np.int32)
        self.written_at = np.zeros(cap, np.float64)
        self.pa = np.zeros((self.acap, ACT_BYTES), np.uint8)
        self.pp = np.zeros(self.acap, np.float16)
        self.rows = self.acts = self.lo = 0

    _ROW_FIELDS = ("x", "z", "astart", "alen", "written_at")

    def state_dict(self):
        ring = np.arange(self.lo, self.rows, dtype=np.int64) % self.cap
        state = {name: getattr(self, name)[ring].copy() for name in self._ROW_FIELDS}
        base = int(state["astart"].min()) if len(ring) else self.acts
        state["astart"] -= base
        at = np.arange(base, self.acts, dtype=np.int64) % self.acap
        state.update(pa=self.pa[at].copy(), pp=self.pp[at].copy())
        state = {name: torch.from_numpy(value) for name, value in state.items()}
        state.update(rows=self.rows, acts=self.acts, lo=self.lo)
        return state

    def load_state_dict(self, state):
        self.lo, self.rows, self.acts = int(state["lo"]), int(state["rows"]), int(state["acts"])
        ring = np.arange(self.lo, self.rows, dtype=np.int64) % self.cap
        for name in self._ROW_FIELDS:
            getattr(self, name)[ring] = state[name].numpy()
        base = self.acts - len(state["pa"])
        at = (np.arange(len(state["pa"]), dtype=np.int64) + base) % self.acap
        self.pa[at], self.pp[at] = state["pa"].numpy(), state["pp"].numpy()
        self.astart[ring] += base

    def add(self, rows, desc, off, target, z):
        n, na = len(rows), len(desc)
        while self.lo < self.rows and (self.rows - self.lo + n > self.cap
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

    def sample_ids(self, batch, rng, recent_mix=0.0, recent_frac=0.2):
        ids = rng.integers(self.lo, self.rows, size=batch)
        k = int(batch * recent_mix)
        if k > 0:
            span = max(1, int((self.rows - self.lo) * recent_frac))
            ids[:k] = rng.integers(self.rows - span, self.rows, size=k)
        return ids


def fuse(net):
    net.evaluate = torch.compile(net.evaluate, dynamic=True)
    return net


def bins(y):
    edges = torch.linspace(-1.0, 1.0, BINS + 1, device=y.device)
    cdf = torch.special.ndtr((edges - y.unsqueeze(1)) * (BINS / 1.5))
    return cdf.diff(dim=1) / (cdf[:, -1:] - cdf[:, :1])


def losses(net, x, desc, parow, target, z, wp):
    logits, pieces = net.evaluate(x)
    value_loss = -(bins(z) * logits.log_softmax(-1)).sum(1).mean()
    logit = net.actions(desc, pieces, parow)
    n = x.shape[0]
    top = torch.full((n,), -1e30, device=x.device).scatter_reduce(0, parow, logit.detach(), reduce="amax")
    shifted = logit - top[parow]
    total = torch.zeros(n, device=x.device).index_add_(0, parow, shifted.exp())
    logp = shifted - total[parow].log()
    policy_loss = -(target * logp).sum() / n
    entropy = -(target * target.clamp(min=1e-30).log()).sum() / n
    stats = {"value_loss": value_loss.detach(),
             "policy_loss": policy_loss.detach(),
             "policy_target_entropy": entropy,
             "policy_kl": policy_loss.detach() - entropy,
             "value_sq_err": (net.value(logits.detach()) - z).square().mean()}
    return value_loss + wp * policy_loss, stats


def train_steps(net, acting, opt, buf, steps, batch, rng, device, batch_fn, policy_w,
                recent_mix, recent_frac, deadline):
    acc, done = collections.Counter(), 0
    for _ in range(steps):
        if time.time() >= deadline:
            break
        parts = batch_fn(buf.gather(buf.sample_ids(batch, rng, recent_mix, recent_frac)), rng, device)
        loss, stats = losses(net, *parts, policy_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        stats["grad_norm"] = nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(acting.parameters()), list(net.parameters()), 1.0 - EMA)
        for key, value in stats.items():
            acc[key] = acc[key] + value
        done += 1
    if not done:
        return 0, {}
    return done, dict(zip(acc, (torch.stack([v.float() for v in acc.values()]) / done).tolist()))


def ingest(buf, data):
    rows = np.asarray(data["rows"], np.uint8).reshape(-1, ROW_BYTES)
    if len(rows):
        buf.add(rows, np.asarray(data["desc"], np.uint8).reshape(-1, ACT_BYTES),
                np.asarray(data["off"], np.int64), np.asarray(data["target"], np.float16),
                np.asarray(data["z"], np.float32))
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
            (not torch.cuda.is_available(), "training requires a working GPU")):
        if bad:
            raise SystemExit(why)

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    if args.train_stream_priority < 0:
        stream = torch.cuda.Stream(device=dev, priority=args.train_stream_priority)
        stream.wait_stream(torch.cuda.current_stream(dev))
        torch.cuda.set_stream(stream)

    import gpu_batch
    torch.manual_seed(args.seed)
    value = Net().to(dev)
    acting = copy.deepcopy(value).requires_grad_(False)
    fuse(value)
    opt = adam(value, args.lr, args.wd)
    buf = Buffer(args.cap)
    if checkpoint:
        value.load_state_dict(checkpoint["value"])
        acting.load_state_dict(checkpoint["acting"])
        opt.load_state_dict(checkpoint["optimizer"])
        target_state = checkpoint["target"]
        publish_state(target_state)
        rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        buf.load_state_dict(checkpoint.pop("buffer"))
    else:
        if args.init_weights:
            for net in (value, acting):
                net.load_state_dict(export_weights.load(args.init_weights).state_dict())
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
        "optimizer_rows": 0, "generated_rows": 0, "farm_runs": 0, "totals": {}}
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
            "buffer": buf.state_dict(), "elapsed": float(el), "next_snapshot": float(el + snap_gap),
            "epoch": epoch, "snapshots": snaps, "progress": progress,
            "cfg": dataclasses.asdict(args), "label": label, "git": args.git,
        }
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)
        for old in snaps[:-2]:
            os.remove(f"{args.out}/{old['file']}")
        del snaps[:-2]
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

    def fit(deadline):
        for group in opt.param_groups:
            group["lr"] = scheduled_lr(args.lr, args.lr_final, time.time() - t0, total, args.lr_stable_frac)
        debt = args.replay_ratio * progress["generated_rows"] - progress["optimizer_rows"]
        steps = int(debt // args.batch) if len(buf) >= args.batch else 0
        done, stats = train_steps(value, acting, opt, buf, steps, args.batch, rng, dev, gpu_batch.make_batch,
                                  args.policy_w, args.recent_mix, args.recent_frac, deadline)
        progress["optimizer_rows"] += done * args.batch
        return done, stats

    def report(phase, window, stats, extra):
        games = max(window["games"], 1)
        rec = {"phase": phase, **{k: int(window[k]) for k in GAME_KEYS},
               "rows": int(window["rows"]), "steps": int(window["steps"]),
               "optimizer_rows": progress["optimizer_rows"], "replay_rows": progress["generated_rows"],
               "timeout_frac": round(window["timeouts"] / games, 3), "buf_s": round(buf.span_seconds(), 1),
               "tgt_mean": round(window["z_sum"] / max(window["rows"], 1), 4),
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
        window["z_sum"] += float(np.asarray(data["z"], np.float64).sum())
        for k in GAME_KEYS:
            window[k] += int(data[k])

    if not checkpoint:
        snapshot(0.0, "init")
    window = collections.Counter()
    while time.time() - t0 < warm:
        absorb(window, warchest.warm(args.warm_games, args.seed * 1_000_003 + epoch, temp=args.temp))
        done, stats = fit(t0 + total)
        window["steps"] += done
        report("warm", window, stats, {})
        window.clear()
    if time.time() - t0 < total:
        acting.push()
        target_state = cpu_state(acting)
        farm = warchest.Farm(args.seed + 1_000_003 * progress["farm_runs"], args.games,
                             [int(d) for d in args.gen_devices.split(",")], sims=args.sims,
                             batch=args.search_batch, puct=args.puct, explore=args.explore)
        progress["farm_runs"] += 1
        next_target = time.time() + args.target_every * 60.0
        next_report = time.time() + 10.0
        at, seen = time.time(), {"gpu_rows": 0, "gpu_calls": 0, "gpu_nanos": 0, "decisions": 0}
        stats = {}
        while time.time() - t0 < total:
            data = farm.collect(args.collect_rows)
            absorb(window, data)
            done, fitted = fit(t0 + total)
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
            gpu = {k: int(data[k]) - seen[k] for k in ("gpu_rows", "gpu_calls", "gpu_nanos")}
            extra = {"evals_per_s": round(gpu["gpu_rows"] / span, 1),
                     "sims_per_s": round(window["rows"] * args.sims / span, 1),
                     "rows_per_s": round(window["rows"] / span, 1),
                     "rows_per_call": round(gpu["gpu_rows"] / max(gpu["gpu_calls"], 1), 1),
                     "gpu_busy": round(1e-9 * gpu["gpu_nanos"] / span / (warchest.PIPELINE * len(args.gen_devices.split(","))), 3)}
            seen.update({k: int(data[k]) for k in ("gpu_rows", "gpu_calls", "gpu_nanos")})
            report("search", window, stats, extra)
            window.clear()
            at, next_report = now, now + 10.0
        del farm
    snapshot(time.time() - t0, "final")


if __name__ == "__main__":
    main()
