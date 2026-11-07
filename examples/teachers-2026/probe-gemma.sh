#!/usr/bin/env bash
# The probe (label --probe) on Gemma 4 12B with 16 similar examples, the best single teacher on CLINC150 and
# Danish MASSIVE in similar.sh, against the same teacher without it. The texts are scored once more, to record
# their states. Pass task names to run only those: probe-gemma.sh clinc150
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
out=examples/teachers-2026/probe
mkdir -p "$out"
common=(--cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --load-4bit
  --teacher google/gemma-4-12B-it --few-shot 8 --few-shot-similar 16 --gold-per-class 8 --probe
  --teacher-devices cuda)
tasks=" ${*:-clinc150 massive-da} "

run_task() {  # name, then the task's own arguments
  local name=$1; shift
  [[ $tasks == *" $name "* ]] || return 0
  local run=$name-gemma4-12b-similar16-probe
  tinydistill run "$@" "${common[@]}" --out "runs/$run"
  cp "runs/$run/results.json" "$out/$run.json"
  cp "runs/$run/results.md" "$out/$run.md"
  tinydistill compare "runs/$name-gemma4-12b-similar16" "runs/$run" > "$out/$name-gemma4-12b.md"
}

run_task clinc150 examples/clinc150/task.yaml --data hf:clinc/clinc_oos:plus:train \
  --test hf:clinc/clinc_oos:plus:test --shortlist 20 --shortlist-keep oos --val-share 0.05
run_task massive-da examples/massive-da/task.yaml --data hf:mteb/amazon_massive_intent:da:train \
  --test hf:mteb/amazon_massive_intent:da:test
