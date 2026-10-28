import json
import shutil
import sys
from pathlib import Path

bot = Path(sys.argv[1])
for sims in map(int, sys.argv[2:]):
    copy = bot.with_name(f"{bot.name}@{sims}")
    shutil.copytree(bot, copy, dirs_exist_ok=True)
    manifest = json.loads((copy / "bot.json").read_text())
    manifest["name"] = f"{manifest['name']}@{sims}"
    manifest["search"]["sims"] = sims
    (copy / "bot.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"packed {copy}", flush=True)
