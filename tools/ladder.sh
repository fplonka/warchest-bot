#!/usr/bin/env bash
set -uo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
campaign=${1:?usage: ladder.sh <campaign> <hours> <devices>}
hours=${2:?usage: ladder.sh <campaign> <hours> <devices>}
devices=${3:?usage: ladder.sh <campaign> <hours> <devices>}
export WARCHEST_BOX_DIR=${WARCHEST_BOX_DIR:-/workspace/curve}
deadline=$(( $(date +%s) + hours * 3600 ))
python=$here/.venv/bin/python

while [ "$(date +%s)" -lt "$deadline" ]; do
    read -r a b games seed kind name_a name_b < <("$python" "$here/tools/ladder.py" "$campaign") || break
    echo "[$(date +%FT%T)] $kind: $name_a vs $name_b (seed $seed)"
    "$here/tools/box.sh" "target/release/ladder $a $b $games $seed $devices 2>/workspace/logs/m$seed.log" \
        | grep '^{' >> "$campaign/matches.jsonl" || { echo "  match failed; see /workspace/logs/m$seed.log"; break; }
    "$python" "$here/tools/ladder.py" "$campaign" report | head -8
done
