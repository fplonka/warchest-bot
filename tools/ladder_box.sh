#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
campaign=$here/${1:?usage: ladder_box.sh <campaign> <hours>}
hours=${2:?usage: ladder_box.sh <campaign> <hours>}
key=$HOME/.ssh/id_ed25519_warchest_vast
vast() { VAST_API_KEY=$(cat ~/.config/vastai/vast_api_key) uvx --from vastai vastai "$@" --raw; }
field() { python3 -c "import json,sys; i=json.load(sys.stdin); print(i$1)"; }

read -r trainer_host trainer_port < "$(git -C "$here" rev-parse --path-format=absolute --git-common-dir)/box"
trainer="ssh -i $key -p $trainer_port -o StrictHostKeyChecking=no -o LogLevel=ERROR"
held=$($trainer "root@$trainer_host" "cd /workspace/warchest && ls -d $(tr '\n' ' ' < "$campaign/bots")" 2>/dev/null || true)
for bot in $held; do
    mkdir -p "$here/$bot"
    rsync -az -e "$trainer" "root@$trainer_host:/workspace/warchest/$bot/" "$here/$bot/"
done

offer=$(vast search offers 'verified=true rentable=true num_gpus=2 gpu_name=RTX_3090 cpu_cores_effective>=12 disk_space>=40 inet_down>=200 reliability>0.98 cuda_vers>=12.8 geolocation in [US,CA,GB,DE,FR,NL,SE,FI,NO,PL,EE,CZ,AT,CH]' -o dph |
    python3 -c 'import json,random,sys; print(random.choice([o["id"] for o in json.load(sys.stdin) if "AMD" in o["cpu_name"]][:3]))')
id=$(vast create instance "$offer" --image nvidia/cuda:12.8.0-devel-ubuntu22.04 --disk 20 --ssh --direct | field '["new_contract"]')
trap 'yes | vast destroy instance "$id"' EXIT
echo "instance $id on offer $offer"
until read -r host port < <(vast show instance "$id" | field '["ssh_host"], i["ssh_port"]') && [ "$host" != None ]; do sleep 15; done
export WARCHEST_BOX_HOST=$host WARCHEST_BOX_PORT=$port WARCHEST_BOX_DIR=/workspace/curve
for _ in $(seq 80); do "$here/tools/box.sh" true 2>/dev/null && break; sleep 15; done
"$here/tools/box.sh" "apt-get update -qq && apt-get install -y -qq rsync libjemalloc2 build-essential curl >/dev/null
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null"
"$here/tools/box.sh" sync
"$here/tools/box.sh" "cargo build --release --bin ladder >/tmp/ladder.log 2>&1 || { tail -30 /tmp/ladder.log; exit 1; }"
(cd "$here" && rsync -azR -e "ssh -i $key -p $port -o StrictHostKeyChecking=no" \
    $(cat "$campaign/bots") "root@$host:/workspace/warchest/")
"$here/tools/ladder.sh" "$campaign" "$hours" 0,1
"$here/tools/box.sh" "grep -hv '^{' /workspace/logs/m*.log | tail -30"
"$here/.venv/bin/python" "$here/tools/ladder.py" "$campaign" report
