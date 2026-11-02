#!/usr/bin/env bash
# Reproduces the financial-tweets results in the README: the 1.5B teacher zero-shot and with 8 gold examples
# per class in its prompt (also used in the student's loss, plus a gold-only baseline); 3 training seeds and
# an upper bound in both. The dataset's validation split is the test set. Labelling and training run on the
# GPU, student benchmarks on the CPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
data=hf:zeroshot/twitter-financial-news-sentiment
base=(examples/financial-tweets/task.yaml --data "$data:train" --test "$data:validation" --cache-dir runs/cache
      --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --upper-bound)

tinydistill run "${base[@]}" --out runs/financial-tweets-1.5b --teacher-devices cpu,cuda
tinydistill run "${base[@]}" --few-shot 8 --gold-per-class 8 --gold-only --out runs/financial-tweets-1.5b-8shot \
  --teacher-devices cpu,cuda

for run in financial-tweets-1.5b financial-tweets-1.5b-8shot; do
  cp "runs/$run/results.json" "examples/financial-tweets/$run.json"
  cp "runs/$run/results.md" "examples/financial-tweets/$run.md"
done
tinydistill compare runs/financial-tweets-1.5b runs/financial-tweets-1.5b-8shot > examples/financial-tweets/teachers.md
python examples/update_readme.py
