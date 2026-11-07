#!/usr/bin/env bash
# The two strongest newer teachers, each shown every text's 16 most similar gold examples (8 gold per class to pick
# from, the same gold the students also train on), on Banking77, CLINC150 and Danish MASSIVE; then both together as
# an ensemble, read from the score cache the single-teacher runs filled (no new labelling). Financial tweets too.
# Pass task names to run only those: similar.sh financial-tweets
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
out=examples/teachers-2026/similar
mkdir -p "$out"
qwen=Qwen/Qwen3.5-9B
gemma=google/gemma-4-12B-it
common=(--cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --load-4bit
  --few-shot 8 --few-shot-similar 16 --gold-per-class 8 --teacher-devices cuda)

tasks=" ${*:-banking77 clinc150 massive-da financial-tweets} "

run_task() {  # name, then the task's own arguments
  local name=$1; shift
  [[ $tasks == *" $name "* ]] || return 0
  for pair in "qwen35-9b:$qwen" "gemma4-12b:$gemma" "ensemble:$qwen,$gemma"; do
    local tag=${pair%%:*} teacher=${pair#*:}
    tinydistill run "$@" "${common[@]}" --teacher "$teacher" --out "runs/$name-$tag-similar16"
    cp "runs/$name-$tag-similar16/results.json" "$out/$name-$tag.json"
    cp "runs/$name-$tag-similar16/results.md" "$out/$name-$tag.md"
  done
  tinydistill compare "runs/$name-qwen35-9b-similar16" "runs/$name-gemma4-12b-similar16" \
    "runs/$name-ensemble-similar16" > "$out/$name.md"
}

run_task banking77 examples/banking77/task.yaml --data hf:PolyAI/banking77:train \
  --test hf:PolyAI/banking77:test --epochs 10
run_task clinc150 examples/clinc150/task.yaml --data hf:clinc/clinc_oos:plus:train \
  --test hf:clinc/clinc_oos:plus:test --shortlist 20 --shortlist-keep oos --val-share 0.05
run_task massive-da examples/massive-da/task.yaml --data hf:mteb/amazon_massive_intent:da:train \
  --test hf:mteb/amazon_massive_intent:da:test
ft=hf:zeroshot/twitter-financial-news-sentiment
run_task financial-tweets examples/financial-tweets/task.yaml --data $ft:train --test $ft:validation --epochs 5
