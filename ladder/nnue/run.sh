#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
lane=$1
C=/workspace/campaigns/nnue
mkdir -p $C
cp ladder/nnue/bots $C/bots
while read -r a b games seed kind na nb < <(flock $C/lock python tools/ladder.py $C); do
    seed=$(( seed * 4 + lane ))
    echo "[$(date +%T)] lane $lane $kind $na vs $nb seed $seed"
    target/release/ladder $a $b $games $seed 2>>$C/err.$lane.log | grep "^{" > $C/out.$lane
    flock $C/lock sh -c "cat $C/out.$lane >> $C/matches.jsonl"
done
