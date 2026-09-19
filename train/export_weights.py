
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from value_net import Net


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = Net()
    net.load_state_dict(ck["acting"])
    return net


def write_bin(net, path):
    with open(path, "wb") as f:
        for a in net.flat():
            a = np.ascontiguousarray(a, np.float32)
            f.write(struct.pack("<I", a.size))
            f.write(a.tobytes())


BOT = Path(__file__).resolve().parents[1] / "engine/target/release/bot"
SEARCH = {"s": 1024, "c": 4.0, "batch": 16, "rounds": 1,
          "puct": 1.5, "prior_temp": 1.0, "cfr": "dcfr"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bot(run, label, net, git, minutes):
    run = Path(run)
    directory = run / "bots" / label
    directory.mkdir(parents=True, exist_ok=True)
    write_bin(net, directory / "weights.bin")
    shutil.copy2(BOT, directory / "bot")
    (directory / "bot.json").write_text(json.dumps({
        "format": 1,
        "name": f"{run.name}.{label}",
        "sha": git,
        "binary": digest(directory / "bot"),
        "weights": "weights.bin",
        "weights_sha": digest(directory / "weights.bin"),
        "search": SEARCH,
        "minutes": round(minutes, 3),
        "note": f"{run.name} {label}, {minutes:.0f} min",
    }, indent=1) + "\n")
    print(f"packed {directory}", flush=True)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    write_bin(load(src), dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
