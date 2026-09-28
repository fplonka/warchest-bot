
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


def read_bin(path):
    raw, arrays, at = Path(path).read_bytes(), [], 0
    for _ in range(3):
        n = struct.unpack_from("<I", raw, at)[0]
        arrays.append(np.frombuffer(raw, np.float32, n, at + 4))
        at += 4 + 4 * n
    assert at == len(raw)
    return Net().load_flat(arrays)


def write_bin(net, path):
    with open(path, "wb") as f:
        for a in net.flat():
            a = np.ascontiguousarray(a, np.float32)
            f.write(struct.pack("<I", a.size))
            f.write(a.tobytes())


BOT = Path(__file__).resolve().parents[1] / "engine/target/release/bot"
SEARCH = {"mind": "search", "sims": 32}
GREEDY = {**SEARCH, "mind": "greedy"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack(directory, name, net, git, search, minutes, note):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_bin(net, directory / "weights.bin")
    shutil.copy2(BOT, directory / "bot")
    (directory / "bot.json").write_text(json.dumps({
        "format": 1,
        "name": name,
        "sha": git,
        "binary": digest(directory / "bot"),
        "weights": "weights.bin",
        "weights_sha": digest(directory / "weights.bin"),
        "search": search,
        "minutes": round(minutes, 3),
        "note": note,
    }, indent=1) + "\n")
    print(f"packed {directory}", flush=True)


def bot(run, label, net, git, minutes):
    run = Path(run)
    pack(run / "bots" / label, f"{run.name}.{label}", net, git, SEARCH, minutes,
         f"{run.name} {label}, {minutes:.0f} min")


def main():
    if sys.argv[1] == "greedy":
        pack(sys.argv[2], "greedy.static", Net(), "static", GREEDY, 0.0, "static-evaluation softmax")
        return
    src, dst = sys.argv[1], sys.argv[2]
    write_bin(load(src), dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
