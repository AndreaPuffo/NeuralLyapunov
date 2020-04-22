#!/usr/bin/env bash

MIN_EXP=1
MAX_EXP=10
MIN_DIM=2
MAX_DIM=10

mkdir logs 2>/dev/null || true
echo "Running for dim [$MIN_DIM, $MAX_DIM] and exp [$MIN_EXP, $MAX_EXP]" >&2
pids=
for n in $(seq $MIN_DIM $MAX_DIM); do
  for e in $(seq $MIN_EXP $MAX_EXP); do
    out="./logs/dim_${n}_exp_${e}.$(date +%s).log"
    python2 -m src.main -e $e -n $n >"$out" 2>&1 &
    pids="$! $pids"
  done
done

echo "PIDs: $pids" >./logs/$(date +%s).pids

