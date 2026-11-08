#!/usr/bin/env bash
# The trimmed multilingual student of trim.sh on the labels of the most accurate Danish teachers so far, each shown
# every text's 16 most similar gold examples: Qwen3.5-9B and Gemma 4 12B as an ensemble
# (examples/teachers-2026/similar.sh), and Gemma 4 12B with the probe (examples/teachers-2026/probe-gemma.sh).
# Their runs fill the score cache, so nothing is labelled here.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
data=hf:mteb/amazon_massive_intent:da
multi=paraphrase-multilingual-MiniLM-L12-v2
student() {  # name suffix, then the teacher's arguments
  local name=$multi-trimmed-$1; shift
  tinydistill run examples/massive-da/task.yaml --data "$data:train" --test "$data:test" --cache-dir runs/cache \
    --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --load-4bit --few-shot 8 --few-shot-similar 16 \
    --gold-per-class 8 "$@" --student "sentence-transformers/$multi" --trim-vocab \
    --out "runs/massive-da-students/$name" --skip-teacher-bench
  cp "runs/massive-da-students/$name/results.json" "examples/massive-da/$name.json"
  cp "runs/massive-da-students/$name/results.md" "examples/massive-da/$name.md"
}
student ensemble-similar16 --teacher Qwen/Qwen3.5-9B,google/gemma-4-12B-it
student gemma4-12b-similar16-probe --teacher google/gemma-4-12B-it --probe
tinydistill compare runs/massive-da-ensemble-similar16 runs/massive-da-students/$multi-trimmed \
  runs/massive-da-students/$multi-trimmed-ensemble-similar16 \
  runs/massive-da-students/$multi-trimmed-gemma4-12b-similar16-probe > examples/massive-da/ensemble-multilingual.md
