#!/usr/bin/env bash
# Reproduces the GoEmotions results in the README: a multi-label task (one yes/no question per label) with
# Qwen2.5-1.5B as the teacher, on the first 10,000 training comments; 8 gold labels per label in the student's
# loss, a gold-only baseline, an upper bound and 3 training seeds. Labelling and training on the GPU,
# benchmarks on the CPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
data=hf:google-research-datasets/go_emotions:simplified

tinydistill run examples/go-emotions/task.yaml --data "$data:train" --limit 10000 --test "$data:test" \
  --cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --upper-bound \
  --gold-per-class 8 --gold-only --epochs 5 --out runs/go-emotions-1.5b --teacher-devices cpu,cuda

cp runs/go-emotions-1.5b/results.json examples/go-emotions/go-emotions-1.5b.json
cp runs/go-emotions-1.5b/results.md examples/go-emotions/go-emotions-1.5b.md
python examples/update_readme.py
