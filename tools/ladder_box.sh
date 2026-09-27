#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
campaign=$here/${1:?usage: ladder_box.sh <campaign> <hours>}
hours=${2:?usage: ladder_box.sh <campaign> <hours>}
key=$HOME/.ssh/id_ed25519_warchest_vast
ssh-add -q "$key"
vast() { VAST_API_KEY=$(cat ~/.config/vastai/vast_api_key) uvx --from vastai vastai "$@" --raw; }
field() { python3 -c "import json,sys; i=json.load(sys.stdin); print(i$1)"; }

runs=$(cut -d/ -f2 "$campaign/bots" | sort -u)
for run in $runs; do
    mkdir -p "$here/runs/$run/bots"
    rsync -az --include '*/' --include bot.json --exclude '*' -e "ssh -i $key -p 17198 -o StrictHostKeyChecking=no" \
        "root@ssh2.vast.ai:/workspace/warchest-engine/runs/$run/bots/" "$here/runs/$run/bots/"
done

offer=$(vast search offers 'verified=true rentable=true num_gpus=2 gpu_name=RTX_3090 cpu_cores_effective>=12 disk_space>=40 inet_down>=200 reliability>0.98 cuda_vers>=12.8 geolocation in [US,CA,GB,DE,FR,NL,SE,FI,NO,PL,EE,CZ,AT,CH]' -o dph |
    python3 -c 'import json,sys; print(next(o["id"] for o in json.load(sys.stdin) if "AMD" in o["cpu_name"]))')
id=$(vast create instance "$offer" --image nvidia/cuda:12.8.0-devel-ubuntu22.04 --disk 20 --ssh --direct | field '["new_contract"]')
trap 'yes | vast destroy instance "$id"' EXIT
echo "instance $id on offer $offer"
until read -r host port < <(vast show instance "$id" | field '["ssh_host"], i["ssh_port"]') && [ "$host" != None ]; do sleep 15; done
export WARCHEST_BOX_HOST=$host WARCHEST_BOX_PORT=$port WARCHEST_BOX_DIR=/workspace/curve
for _ in $(seq 80); do "$here/tools/box.sh" true 2>/dev/null && break; sleep 15; done
"$here/tools/box.sh" "apt-get update -qq && apt-get install -y -qq rsync libjemalloc2 build-essential curl >/dev/null
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null"
"$here/tools/box.sh" sync
"$here/tools/box.sh" "cd engine && cargo build --release --bin ladder >/tmp/ladder.log 2>&1 || { tail -30 /tmp/ladder.log; exit 1; }"
read -r ip direct < <(vast show instance "$id" | field '["public_ipaddr"], i["ports"]["22/tcp"][0]["HostPort"]')
ssh -A -i "$key" -p 17198 -o StrictHostKeyChecking=no root@ssh2.vast.ai \
    "cd /workspace/warchest-engine && rsync -azR -e 'ssh -p $direct -o StrictHostKeyChecking=no' \
     $(tr '\n' ' ' < "$campaign/bots") root@$ip:/workspace/warchest-engine/"
"$here/tools/box.sh" "sed -i 's/\"s\": 1024/\"s\": 256/' /workspace/warchest-engine/runs/*/bots/*/bot.json"
"$here/tools/ladder.sh" "$campaign" "$hours"
"$here/tools/box.sh" "grep -hv '^{' /workspace/logs/m*.log | tail -30"
"$here/.venv/bin/python" "$here/tools/ladder.py" "$campaign" report
