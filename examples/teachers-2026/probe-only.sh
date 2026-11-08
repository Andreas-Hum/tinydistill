#!/usr/bin/env bash
# The probe alone as the teacher (label --probe-only): no label is scored, only each text's state, so labelling
# is much faster. Each task's best probe teacher, with a fixed zero-shot prompt (cached once for every text) and
# with each text's 16 most similar gold examples, fit on the same 8 gold examples per class. Against the runs of
# probe.sh and probe-gemma.sh, which also score the labels in context. Pass task names to run only those.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
out=examples/teachers-2026/probe-only
mkdir -p "$out"
common=(--cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --load-4bit
  --gold-per-class 8 --probe-only --teacher-devices cuda)
tasks=" ${*:-banking77 clinc150 massive-da} "

run_task() {  # name, tag, teacher, then the task's own arguments
  local name=$1 tag=$2 teacher=$3; shift 3
  [[ $tasks == *" $name "* ]] || return 0
  local runs=()
  for prompt in zeroshot similar16; do
    local run=$name-$tag-probe-only-$prompt extra=()
    [[ $prompt == similar16 ]] && extra=(--few-shot 8 --few-shot-similar 16)
    tinydistill run "$@" "${common[@]}" --teacher "$teacher" "${extra[@]}" --out "runs/$run"
    cp "runs/$run/results.json" "$out/$run.json"
    cp "runs/$run/results.md" "$out/$run.md"
    runs+=("runs/$run")
  done
  tinydistill compare "${runs[@]}" > "$out/$name.md"
}

run_task banking77 7b Qwen/Qwen2.5-7B-Instruct examples/banking77/task.yaml --data hf:PolyAI/banking77:train \
  --test hf:PolyAI/banking77:test --epochs 10
run_task clinc150 gemma4-12b google/gemma-4-12B-it examples/clinc150/task.yaml \
  --data hf:clinc/clinc_oos:plus:train --test hf:clinc/clinc_oos:plus:test --shortlist 20 --shortlist-keep oos \
  --val-share 0.05
run_task massive-da gemma4-12b google/gemma-4-12B-it examples/massive-da/task.yaml \
  --data hf:mteb/amazon_massive_intent:da:train --test hf:mteb/amazon_massive_intent:da:test \
  --student sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --trim-vocab
