#!/usr/bin/env bash
# The trimmed multilingual student of trim.sh on the labels of the most accurate Danish teacher so far: Qwen3.5-9B
# and Gemma 4 12B as an ensemble, each shown every text's 16 most similar gold examples (examples/teachers-2026/
# similar.sh, whose runs fill the score cache, so nothing is labelled here).
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
data=hf:mteb/amazon_massive_intent:da
name=paraphrase-multilingual-MiniLM-L12-v2-trimmed-ensemble-similar16
out=runs/massive-da-students/$name
tinydistill run examples/massive-da/task.yaml --data "$data:train" --test "$data:test" --cache-dir runs/cache \
  --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --load-4bit \
  --teacher Qwen/Qwen3.5-9B,google/gemma-4-12B-it --few-shot 8 --few-shot-similar 16 --gold-per-class 8 \
  --student sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --trim-vocab --out "$out" \
  --skip-teacher-bench
cp "$out/results.json" "examples/massive-da/$name.json"
cp "$out/results.md" "examples/massive-da/$name.md"
tinydistill compare runs/massive-da-ensemble-similar16 runs/massive-da-students/paraphrase-multilingual-MiniLM-L12-v2-trimmed \
  "$out" > examples/massive-da/ensemble-multilingual.md
