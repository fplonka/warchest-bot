#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob
cd "$(dirname "$0")/../.."
lane=$1
C=runs/1028-ladder
mkdir -p $C
cp ladder/nnue/bots $C/bots
while :; do
    for spec in runs/1028-nnue-*/bots/w*k/bot.json; do
        bot=$(dirname "$spec")
        [ -e "$bot@256" ] || flock $C/lock python ladder/nnue/at_sims.py "$bot" 256 1024 4096
    done
    read -r a b games seed kind na nb < <(flock $C/lock python tools/ladder.py $C) || break
    seed=$(( seed * 4 + lane ))
    echo "[$(date +%T)] lane $lane $kind $na vs $nb seed $seed"
    target/release/ladder $a $b $games $seed 2>>$C/err.$lane.log | grep "^{" > $C/out.$lane
    flock $C/lock sh -c "cat $C/out.$lane >> $C/matches.jsonl && python tools/ladder.py $C json 1028-teacher.s256"
done
