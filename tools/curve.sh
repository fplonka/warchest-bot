#!/usr/bin/env bash
set -uo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export WARCHEST_BOX_DIR=/workspace/curve WARCHEST_BOX_LOCAL_DIR=$here
queue=$here/runs/curve/queue
mkdir -p "$here/runs/curve"

run() {
    local branch=$1 tree=$here/../curve-$branch
    shift
    git -C "$here" worktree add --force --detach "$tree" "$branch" &&
        "$tree/tools/box.sh" go --skip-ladder "$@"
    local status=$?
    git -C "$here" worktree remove --force "$tree"
    return $status
}

candidate() {
    local branch=$1 minutes=$2
    "$here/tools/box.sh" "rm -rf runs/smoke_$branch" &&
        run "$branch" "out=${branch}-${minutes}m_s101" minutes="$minutes" \
            snapshot_every=$((minutes / 4)) seed=101
}

while :; do
    read -r branch minutes < <(head -1 "$queue" 2>/dev/null)
    minutes=${minutes:-120}
    if [ -z "$branch" ]; then
        sleep 60
        continue
    fi
    if ! "$here/tools/box.sh" true >/dev/null 2>&1; then
        echo "[$(date +%FT%T)] box unreachable, waiting"
        sleep 120
        continue
    fi
    echo "[$(date +%FT%T)] $branch ${minutes}min"
    candidate "$branch" "$minutes" || echo "[$(date +%FT%T)] $branch failed"
    tail -n +2 "$queue" > "$queue.tmp" && mv "$queue.tmp" "$queue"
done
