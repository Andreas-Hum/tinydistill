#!/usr/bin/env bash
# Reproduces the Banking77 results in the README. The teacher labels and the student trains on the GPU;
# every student benchmark runs on the CPU. The teacher is timed on the CPU (the speed-up reference) and the GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-runs/banking77}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

tinydistill run examples/banking77/task.yaml \
  --data hf:PolyAI/banking77:train \
  --test hf:PolyAI/banking77:test \
  --out "$OUT" --device cuda --threads "$OMP_NUM_THREADS" --int8 \
  --teacher-devices cpu,cuda --teacher-reps 5 "$@"

cp "$OUT/results.json" "$OUT/results.md" examples/banking77/
python examples/banking77/update_readme.py
