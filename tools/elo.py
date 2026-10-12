import json
import math
import sys
from pathlib import Path
from statistics import linear_regression

from ladder import ratings

campaign = Path(sys.argv[1])
bots, names, _, mean, covariance = ratings(campaign)
starts = json.loads((campaign / "stages.json").read_text())
block = covariance[:len(names), :len(names)]
spread = (block - block.mean(0) - block.mean(1, keepdim=True) + block.mean()).diagonal().sqrt()
run, anchors = [], {}
for i, name in enumerate(names):
    stage = name.split(".")[0]
    if stage in starts:
        run.append({"name": name, "hours": starts[stage] + bots[name][0] / 60,
                    "elo": float(mean[i]), "se": float(spread[i])})
    else:
        anchors[name] = float(mean[i])
run.sort(key=lambda bot: bot["hours"])
slope, intercept = linear_regression([math.log2(bot["hours"]) for bot in run],
                                     [bot["elo"] for bot in run])
report = json.dumps({"bots": run, "anchors": anchors, "switches": sorted(starts.values())[1:],
                     "fit": {"a": intercept, "b": slope}})
for stage in (campaign / "stages").iterdir():
    (stage / "elo.json").write_text(report)
