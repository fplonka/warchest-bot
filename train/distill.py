import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import export_weights
import gpu_batch
import nnue
import warchest
from train import HOLDOUT_ROWS, Buffer, losses

CHUNK = 32768


@torch.no_grad()
def label(teacher, buf, device):
    win = []
    for ids in np.array_split(np.arange(buf.lo, buf.rows), -(-len(buf) // CHUNK)):
        with torch.autocast("cuda", torch.bfloat16):
            win.append(teacher.evaluate(gpu_batch.make_batch(buf.gather(ids), device)[0])[0].float().sigmoid().cpu())
    return torch.cat(win).numpy()


def student_stats(student, parts, device):
    rows, desc, parow, target, z = parts
    win, logp = student(*nnue.inputs(rows, desc, parow, device))
    t = lambda a: torch.as_tensor(a, dtype=torch.float32).to(device, non_blocking=True)
    target, result = t(target), (t(z[:, 0]) + 1) / 2
    policy_loss = -(target * logp).sum() / target.sum()
    entropy = -(target * target.clamp(min=1e-30).log()).sum() / target.sum()
    return win, result, {"value_loss": F.binary_cross_entropy_with_logits(win, result),
                         "policy_loss": policy_loss, "policy_kl": policy_loss - entropy}


@torch.no_grad()
def scores(stats, buf):
    runs = [stats(buf.gather(ids)) for ids in np.array_split(np.arange(buf.lo, buf.rows), -(-len(buf) // CHUNK))]
    return " ".join(f"{k} {np.mean([r[k].item() for r in runs]):.4f}" for k in ("value_loss", "policy_kl"))


def parity(student, buf, device, weights):
    rows, desc, parow, _, _ = buf.gather(np.arange(buf.lo, buf.lo + 2048))
    with torch.no_grad():
        win, logp = student(*nnue.inputs(rows, desc, parow, device))
    off = np.concatenate([[0], np.cumsum(np.bincount(parow, minlength=len(rows)))]).astype(np.uint32)
    value, prior = warchest.nnue_eval(str(weights), rows.ravel(), desc.ravel(), off)
    return (f"rust vs torch max |d|: value {np.abs(np.tanh(win.cpu().numpy() / 2) - value).max():.2e} "
            f"prior {np.abs(logp.exp().cpu().numpy() - prior).max():.2e}")


def main():
    ap = argparse.ArgumentParser(description="Distil a teacher snapshot into an NNUE student over its replay buffer, and pack it as a bot.")
    ap.add_argument("--teacher", required=True, help="a snapshot .pt; its .buffer holds the training rows")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--sims", type=int, required=True)
    ap.add_argument("--lam", type=float, default=1.0, help="value target: lam * teacher + (1 - lam) * game result")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cap", type=int, default=config.Cfg.cap)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0, out, device = time.time(), Path(args.out), torch.device(args.device)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    buf = Buffer(args.cap).load(Path(args.teacher).with_suffix(".buffer"))
    held = Buffer(HOLDOUT_ROWS).load(args.holdout)
    teacher = export_weights.load(args.teacher).to(device).eval()
    print(f"{len(buf)} training rows, {len(held)} held out, loaded in {time.time() - t0:.0f}s", flush=True)
    wins = label(teacher, buf, device)
    teacher_stats = lambda parts: losses(teacher, *gpu_batch.make_batch(parts, device), 1.0)[1]
    print(f"labelled in {time.time() - t0:.0f}s; teacher on held-out rows: {scores(teacher_stats, held)}", flush=True)
    student = nnue.Student().to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
    decay = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1 - step / args.steps)
    for step in range(1, args.steps + 1):
        ids = rng.integers(buf.lo, buf.rows, size=args.batch)
        win, result, stats = student_stats(student, buf.gather(ids), device)
        value = torch.as_tensor(wins[ids - buf.lo], dtype=torch.float32, device=device)
        loss = F.binary_cross_entropy_with_logits(win, args.lam * value + (1 - args.lam) * result) + stats["policy_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        decay.step()
        student.clip()
        if step % max(args.steps // 20, 1) == 0:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s student on held-out rows: "
                  f"{scores(lambda parts: student_stats(student, parts, device)[2], held)}", flush=True)
    torch.save(student.state_dict(), out / "student.pt")
    bot = out / "bots" / "final"
    export_weights.pack(bot, f"{out.name}.final", student, config.git_sha(), {"mind": "nnue", "sims": args.sims},
                        (time.time() - t0) / 60, f"nnue student of {args.teacher}, {args.steps} steps of {args.batch}, lam {args.lam}")
    print(parity(student, held, device, bot / "weights.bin"), flush=True)


if __name__ == "__main__":
    main()
