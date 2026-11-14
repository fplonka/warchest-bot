#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
read -r host port < "$(git -C "$here" rev-parse --path-format=absolute --git-common-dir)/box"
host=${WARCHEST_BOX_HOST:-$host}
port=${WARCHEST_BOX_PORT:-$port}
key=${WARCHEST_BOX_KEY:-$HOME/.ssh/id_ed25519_warchest_vast}
remote=${WARCHEST_BOX_DIR:-/workspace/warchest}
runs=/workspace/warchest/runs
local_dir=${WARCHEST_BOX_LOCAL_DIR:-$here}

ssh_opts=(-i "$key" -p "$port" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o LogLevel=ERROR)
prelude="export PATH=/root/.cargo/bin:/usr/local/cuda/bin:\$PATH
unset CONDA_PREFIX
ulimit -c 0
[ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ] && export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
mkdir -p $remote && cd $remote
[ -f venv/bin/activate ] && source venv/bin/activate"

run_remote() { ssh "${ssh_opts[@]}" "root@$host" "$prelude
$*"; }

build_script="[ -d venv ] || python -m venv --system-site-packages venv
source venv/bin/activate
find warchest engine bot -type f -exec touch {} +
cd bot
maturin develop --release --features python,gpu >maturin.log 2>&1 || { tail -40 maturin.log; exit 1; }
tail -2 maturin.log
cd .. && cargo build --release --features bot/gpu --bin bot --bin ladder >cargo.log 2>&1 || { tail -40 cargo.log; exit 1; }
tail -1 cargo.log"

case "${1:-}" in
setup)
    run_remote "apt-get update -qq && apt-get install -y -qq libjemalloc2 rsync >/dev/null
[ -x /root/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
pip install -q maturin numpy nvidia-ml-py
cargo --version && nvcc --version | tail -1 && python -c 'import torch; print(torch.__version__, torch.cuda.device_count(), \"gpus\")'"
    ;;
sync)
    rsync -az --delete -e "ssh ${ssh_opts[*]}" \
        --exclude runs --exclude bots --exclude arena --exclude suites \
        --exclude target --exclude .venv --exclude papers \
        --exclude __pycache__ --exclude .git --exclude data \
        "$here/" "root@$host:$remote/"
    [ "$remote" = /workspace/warchest ] ||
        run_remote "[ -e runs ] || ln -s $runs runs"
    sha=$(git -C "$here" rev-parse --short=7 HEAD)
    git -C "$here" diff-index --quiet HEAD -- . ':!runs' || sha="${sha}+dirty"
    echo "$sha" | ssh "${ssh_opts[@]}" "root@$host" "cat > $remote/.gitsha"
    echo "synced -> $host:$remote"
    ;;
pull)
    name=${2:-}
    filters=(--exclude '*.tmp' --exclude 'snap_*'
             --exclude 'bots/*/bot' --exclude 'bots/*/weights.bin')
    if [ -z "$name" ]; then
        filters+=(--include '*/' --include '*.html' --include 'plotly.min.js'
                  --include 'log.json' --include 'epochs.jsonl' --include 'bot.json'
                  --include 'ladder.json' --include 'matches.jsonl' --include 'bots' --include 'progress.jsonl' --include 'elo.json' --include 'comparisons/*.json'
                  --include 'config.json' --include 'NOTES.md'
                  --include 'train.log' --exclude '*')
    fi
    mkdir -p "$local_dir/runs${name:+/$name}"
    rsync -az --timeout=60 -e "ssh ${ssh_opts[*]}" "${filters[@]}" \
        "root@$host:$remote/runs/$name/" "$local_dir/runs${name:+/$name}/"
    if [ -z "$name" ]; then
        run_remote "find /workspace -maxdepth 5 -path '*/runs/*/epochs.jsonl' -mmin -2880 -printf '%T@ %h\n'" |
            sort -rn | awk -v skip="$remote/runs/" 'index($2, skip) != 1 { n = split($2, p, "/"); if (!seen[p[n]]++) print $2 }' |
            while read -r dir; do
                rsync -azu --timeout=60 -e "ssh ${ssh_opts[*]}" "${filters[@]}" "root@$host:$dir/" "$local_dir/runs/${dir##*/}/" < /dev/null
            done
    fi
    echo "pulled ${name:-runs}"
    ;;
build)
    "$0" start build bash -c "$build_script"
    "$0" follow build
    ;;
follow)
    tag=${2:?usage: follow <tag> [run]}
    if [ -n "${3:-}" ]; then
        (while :; do
            "$0" pull "$3" >/dev/null 2>&1 || true
            sleep "${WARCHEST_BOX_POLL:-60}"
        done) &
        pull_pid=$!
        trap 'kill "$pull_pid" 2>/dev/null || true; wait "$pull_pid" 2>/dev/null || true' EXIT
    fi
    while run_remote "while [ ! -s /workspace/logs/$tag.pid ] || kill -0 \$(cat /workspace/logs/$tag.pid) 2>/dev/null; do
    tail -1 /workspace/logs/$tag.log 2>/dev/null || true
    sleep \${WARCHEST_BOX_POLL:-60}
done
cat /workspace/logs/$tag.exit
grep -qx 0 /workspace/logs/$tag.exit || { tail -20 /workspace/logs/$tag.log; exit 1; }"; status=$?; [ "$status" -eq 255 ]; do
        echo "connection to $host lost, reconnecting"
        sleep 60
    done
    [ -n "${3:-}" ] && "$0" pull "$3"
    [ "$status" -eq 0 ] && echo "JOB_DONE tag=$tag ok" || {
        echo "JOB_DONE tag=$tag failed"
        exit 1
    }
    ;;
start)
    tag=${2:?usage: start <tag> <command...>}
    shift 2
    "$0" status
    run_remote "mkdir -p /workspace/logs
[ -s /workspace/logs/$tag.pid ] && kill -0 \$(cat /workspace/logs/$tag.pid) 2>/dev/null && { echo \"$tag is still running\"; exit 1; }
rm -f /workspace/logs/$tag.pid /workspace/logs/$tag.exit
echo $remote > /workspace/logs/$tag.owner
cat > /workspace/logs/$tag.sh <<'EOS'
echo \$\$ > /workspace/logs/$tag.pid
$prelude
$(printf '%q ' "$@")
echo \$? > /workspace/logs/$tag.exit
EOS
nohup setsid bash /workspace/logs/$tag.sh >/workspace/logs/$tag.log 2>&1 < /dev/null &
echo started $tag"
    ;;
status)
    ssh "${ssh_opts[@]}" "root@$host" python3 - < "$here/tools/status.py"
    ;;
kill)
    tag=${2:?usage: kill <tag>}
    run_remote "[ \"\$(cat /workspace/logs/$tag.owner 2>/dev/null)\" = $remote ] || { echo \"$tag was not started from $remote; not killed\"; exit 1; }
kill -- -\$(cat /workspace/logs/$tag.pid) && echo killed $tag"
    ;;
run)
    sha=$(git -C "$here" rev-parse --short=7 "${2:?usage: run <commit> <tag> <command...>}^{commit}")
    tag=${3:?usage: run <commit> <tag> <command...>}
    shift 3
    code=/workspace/code/$sha
    run_remote "[ -d $code ]" || git -C "$here" archive "$sha" ':!runs' | ssh "${ssh_opts[@]}" "root@$host" \
        "mkdir -p $code.tmp && tar x -C $code.tmp && echo $sha > $code.tmp/.gitsha && ln -s $runs $code.tmp/runs && mv $code.tmp $code"
    WARCHEST_BOX_DIR=$code "$0" start "$tag" bash -c "($build_script) && source venv/bin/activate && exec $(printf '%q ' "$@")"
    ;;
go)
    commit=${2:?usage: go <commit> out=<name> [knob=value ...]}
    shift 2
    out=
    for a in "$@"; do
        case "$a" in out=*) out=${a#out=} ;; esac
    done
    [ -n "$out" ] || { echo "go needs out=<name>" >&2; exit 1; }
    "$0" run "$commit" "$out" bash -c "python train/check_parity.py && exec bash tools/resume_train.sh $(printf '%q ' "$@")"
    "$0" follow "$out" "$out"
    ;;
"")  sed -n '4,11p' "$0" ;;
*)   run_remote "$*" ;;
esac
