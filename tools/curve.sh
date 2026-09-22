#!/usr/bin/env bash
set -uo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export WARCHEST_BOX_DIR=/workspace/curve WARCHEST_BOX_LOCAL_DIR=$here
queue=$here/runs/curve/queue
campaign=${WARCHEST_CAMPAIGN:-$here/ladder/long}
mkdir -p "$here/runs/curve"

run() {
    local branch=$1 tree=$here/../curve-$branch
    shift
    git -C "$here" worktree add --force --detach "$tree" "$branch" &&
        "$tree/tools/box.sh" go "$@"
    local status=$?
    git -C "$here" worktree remove --force "$tree"
    return $status
}

candidate() {
    local name=$1 branch=$2
    shift 2
    run "$branch" "out=$name" minutes=480 snapshot_every=60 \
        seed=1001 lr_final=0.001 "$@" &&
        "$here/tools/box.sh" pull "$name" &&
        { grep -qxF "runs/$name/bots/*" "$campaign/bots" ||
              echo "runs/$name/bots/*" >> "$campaign/bots"; }
}

while :; do
    read -r name branch rest < <(head -1 "$queue" 2>/dev/null)
    if [ -z "${name:-}" ]; then
        sleep 60
        continue
    fi
    if ! "$here/tools/box.sh" true >/dev/null 2>&1; then
        echo "[$(date +%FT%T)] box unreachable, waiting"
        sleep 120
        continue
    fi
    echo "[$(date +%FT%T)] $name on $branch ${rest:-}"
    # shellcheck disable=SC2086
    candidate "$name" "$branch" $rest || echo "[$(date +%FT%T)] $name failed"
    tail -n +2 "$queue" > "$queue.tmp" && mv "$queue.tmp" "$queue"
    "$here/tools/ladder.sh" "$campaign" 3
done
