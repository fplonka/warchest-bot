#!/usr/bin/env bash
set -uo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
campaign=${1:?usage: ladder.sh <campaign> <hours>}
hours=${2:?usage: ladder.sh <campaign> <hours>}
export WARCHEST_BOX_DIR=/workspace/curve
deadline=$(( $(date +%s) + hours * 3600 ))
python=$here/.venv/bin/python

while [ "$(date +%s)" -lt "$deadline" ]; do
    read -r a b games seed kind name_a name_b < <("$python" "$here/tools/ladder.py" "$campaign") || break
    tag="m$seed"
    echo "[$(date +%FT%T)] $kind: $name_a vs $name_b (seed $seed)"
    "$here/tools/box.sh" start "$tag" \
        engine/target/release/ladder "$a" "$b" "$games" "$seed" || break
    if ! "$here/tools/box.sh" follow "$tag" >/dev/null; then
        echo "  match failed; see /workspace/logs/$tag.log"
        break
    fi
    "$here/tools/box.sh" "grep '^{' /workspace/logs/$tag.log" \
        | grep '^{' >> "$campaign/matches.jsonl" || break
    "$python" "$here/tools/ladder.py" "$campaign" report | head -8
done
