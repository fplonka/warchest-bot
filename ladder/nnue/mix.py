import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train"))

import export_weights

teacher, student, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
for sims in map(int, sys.argv[4:]):
    for value, prior, label in ((teacher, student, "tv-sp"), (student, teacher, "sv-tp")):
        bot = out / f"{label}@{sims}"
        shutil.copytree(student, bot, dirs_exist_ok=True)
        shutil.copy2(export_weights.BOT, bot / "bot")
        manifest = json.loads((bot / "bot.json").read_text())
        manifest.update(name=f"{out.parent.name}.{label}@{sims}", binary=export_weights.digest(bot / "bot"),
                        search={"mind": "mix", "sims": sims, "value": str(value), "prior": str(prior)})
        (bot / "bot.json").write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"packed {bot}", flush=True)
