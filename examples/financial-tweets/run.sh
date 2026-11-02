#!/usr/bin/env bash
# Reproduces the financial-tweets results in the README: pure distillation plus an upper-bound row. The
# dataset's validation split is the test set. Labelling and training run on the GPU, student benchmarks on
# the CPU; the teacher is timed on the CPU and the GPU. Extra arguments are passed on; SUFFIX names outputs.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
SUFFIX=${SUFFIX:-}
data=hf:zeroshot/twitter-financial-news-sentiment

tinydistill run examples/financial-tweets/task.yaml --data "$data:train" --test "$data:validation" \
  --out "runs/financial-tweets$SUFFIX" --cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" \
  --int8 --teacher-devices cpu,cuda --teacher-reps 5 --upper-bound "$@"

cp "runs/financial-tweets$SUFFIX/results.json" "examples/financial-tweets/results$SUFFIX.json"
cp "runs/financial-tweets$SUFFIX/results.md" "examples/financial-tweets/results$SUFFIX.md"
python examples/update_readme.py
