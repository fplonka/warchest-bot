import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

import nnue
import shards
import warchest

bot, data, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
shard = shards.Shard(shards.listed(data, True)[0])
rows, desc, parow, teacher, value = shard.gather(np.arange(min(n, len(shard))))
off = np.concatenate([[0], np.cumsum(np.bincount(parow, minlength=len(rows)))])
student_value, student = warchest.nnue_eval(str(bot / "weights.bin"), rows.ravel(), desc.ravel(), off.astype(np.uint32))
log = lambda p: np.log(np.maximum(p, 1e-12))
kl = teacher * (log(teacher) - log(student))
starts = off[:-1]
student_rank = np.array([(student[a:b] > student[a:b][np.argmax(teacher[a:b])]).sum() for a, b in zip(off[:-1], off[1:])])
print(f"{bot.name}: {len(rows)} held-out positions, policy KL {kl.sum() / len(rows):.3f}, value MSE {np.mean((student_value - value) ** 2):.4f}")
print(f"teacher's best move is the student's top 1 {np.mean(student_rank == 0):.1%}, top 4 {np.mean(student_rank < 4):.1%}, top 8 {np.mean(student_rank < 8):.1%}")
kinds = desc[:, 0]
share = {k: kl[kinds == k].sum() / kl.sum() for k in np.unique(kinds)}
for k, s in sorted(share.items(), key=lambda x: -x[1])[:8]:
    print(f"  {warchest.KIND_NAMES[k]:>16s}: {s:.0%} of the KL, {np.mean(kinds == k):.0%} of the moves")
phases = nnue.inputs(rows, desc, parow, "cpu")[3].numpy()
per = np.add.reduceat(kl, starts)
for lo, hi in ((0, 6), (6, 12), (12, 20), (20, 99)):
    m = (phases >= lo) & (phases < hi)
    print(f"  phase {lo:>2}-{hi:<2}: {np.mean(m):.0%} of positions, KL {per[m].mean():.3f}")
