import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train"))

import export_weights

bot = Path(sys.argv[1])
for sims in map(int, sys.argv[2:]):
    copy = bot.with_name(f"{bot.name}@{sims}")
    shutil.copytree(bot, copy, dirs_exist_ok=True, ignore=shutil.ignore_patterns("student.pt"))
    shutil.copy2(export_weights.BOT, copy / "bot")
    manifest = json.loads((copy / "bot.json").read_text())
    manifest["name"] = f"{manifest['name']}@{sims}"
    manifest["search"]["sims"] = sims
    manifest["binary"] = export_weights.digest(copy / "bot")
    (copy / "bot.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"packed {copy}", flush=True)
