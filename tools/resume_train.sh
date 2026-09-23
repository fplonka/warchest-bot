#!/usr/bin/env bash
set -euo pipefail

out=
for arg in "$@"; do
    case "$arg" in out=*) out=${arg#out=} ;; esac
done
[ -n "$out" ] || { echo "resume_train.sh needs out=<name>" >&2; exit 2; }
dir="runs/${out#runs/}"

newest_snapshot() {
    ls -1t "$dir"/snap_*.pt 2>/dev/null | head -1
}

resume=$(newest_snapshot || true)
failures=0
while :; do
    if python train/train.py "$@" ${resume:+"resume=$resume"}; then
        exit 0
    else
        status=$?
    fi
    next=$(newest_snapshot || true)
    [ -n "$next" ] || exit "$status"
    if [ "$next" = "$resume" ]; then
        failures=$((failures + 1))
        if [ "$failures" -ge 3 ]; then
            echo "[box] trainer failed $failures times without a new snapshot; stopping" >&2
            exit "$status"
        fi
    else
        failures=0
    fi
    resume=$next
    echo "[box] trainer exited $status; resuming $resume" >&2
    sleep 5
done
