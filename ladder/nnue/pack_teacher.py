import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train"))

import config
import export_weights

snapshot, out = sys.argv[1], Path(sys.argv[2])
net = export_weights.load(snapshot)
for sims in map(int, sys.argv[3:]):
    export_weights.pack(out / "bots" / f"s{sims}", f"{out.name}.s{sims}", net, config.git_sha(),
                        {"mind": "search", "sims": sims}, 0.0, f"{snapshot} at {sims} sims, counting inference")
