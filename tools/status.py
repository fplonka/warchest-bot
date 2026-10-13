import glob
import json
import os
import shutil
import subprocess
import time

SHOW = (("value", "value_loss", ".3f"), ("fresh value", "fresh_value_loss", ".3f"),
        ("policy", "policy_loss", ".3f"), ("rows/s", "rows_per_s", ",.0f"), ("buffer", "buf", ",.0f"))
HELD = (("holdout value", "holdout_value_loss", ".3f"), ("holdout KL", "holdout_policy_kl", ".3f"))


def gb(n):
    return f"{n / 1e9:.0f}"


def cgroup(name):
    with open(f"/sys/fs/cgroup/{name}") as f:
        return f.read().strip()


def pairs(name):
    return dict(line.split() for line in cgroup(name).splitlines())


def tail(path):
    with open(path, "rb") as f:
        start = max(0, os.path.getsize(path) - (256 << 10))
        f.seek(start)
        return [json.loads(raw) for raw in f.read().split(b"\n")[bool(start):-1]]


def fields(row, show):
    return "  ".join(f"{label} {row[key]:{spec}}" for label, key, spec in show if key in row)


def machine():
    disk = shutil.disk_usage("/workspace")
    print(f"memory {gb(int(pairs('memory.stat')['anon']))}/{gb(int(cgroup('memory.max')))} GB, "
          f"{pairs('memory.events')['oom_kill']} OOM kills · disk {gb(disk.free)} GB free · "
          f"threads {cgroup('pids.current')}/{cgroup('pids.max')}")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for line in smi.splitlines():
        i, busy, used, total = (int(v) for v in line.split(","))
        print(f"gpu{i} {busy}% busy, {used / 1024:.0f}/{total / 1024:.0f} GB")


def jobs():
    print("\njobs:")
    ps = subprocess.run(["ps", "-eo", "pid=,ppid=,sid=,etime=,args="], capture_output=True, text=True).stdout
    for pid, ppid, sid, up, args in (line.split(None, 4) for line in ps.splitlines()):
        if ppid == "1" and sid != "1" and not args.startswith(("sleep", "sshd")):
            print(f"  {pid:>8}  up {up:>11}  {os.readlink(f'/proc/{pid}/cwd')}  {args[:160]}")


def runs():
    print("\nruns written in the last two days:")
    found = {os.path.realpath(p) for depth in range(4)
             for p in glob.glob(f"/workspace/{'*/' * depth}runs/*/epochs.jsonl")}
    now = time.time()
    for path in sorted(found, key=os.path.getmtime, reverse=True):
        quiet = now - os.path.getmtime(path)
        if quiet > 2 * 86400:
            continue
        run = os.path.dirname(path)
        with open(f"{run}/log.json") as f:
            cfg = json.load(f)["cfg"]
        rows = tail(path)
        search = [r for r in rows if r.get("phase") == "search"]
        snaps = [r for r in rows if r.get("phase") == "snapshot"]
        state = "live" if quiet < 300 else f"quiet for {quiet / 3600:.1f} h"
        print(f"  {run}  {state}  {rows[-1]['t'] / 3600:.1f}/{cfg['minutes'] / 60:.0f} h  git {cfg['git']}")
        if search:
            print(f"      {fields(search[-1], SHOW)}")
        if snaps:
            print(f"      snapshot {snaps[-1]['label']} at {snaps[-1]['t'] / 3600:.1f} h  {fields(snaps[-1], HELD)}")
        if os.path.exists(f"{run}/elo.json"):
            with open(f"{run}/elo.json") as f:
                bots = json.load(f)["bots"]
            print("      elo " + "  ".join(f"{b['name']} {b['elo']:+.0f}±{b['se']:.0f}" for b in bots[-4:]))


machine()
jobs()
runs()
