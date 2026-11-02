#!/usr/bin/env bash
# Reproduces the Banking77 results in the README: pure distillation, and distillation plus 8 gold labels per
# class with an upper-bound row. Labelling and training run on the GPU; every student benchmark runs on the
# CPU. The teacher is timed on the CPU (the speed-up reference) and on the GPU. Both runs share one teacher
# cache, so the texts are labelled once. Extra arguments are passed on (for example another --teacher), and
# SUFFIX names the outputs (results$SUFFIX.md).
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
SUFFIX=${SUFFIX:-}
common=(examples/banking77/task.yaml --data hf:PolyAI/banking77:train --test hf:PolyAI/banking77:test
        --cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" --int8
        --teacher-devices cpu,cuda --teacher-reps 5)

tinydistill run "${common[@]}" --out "runs/banking77$SUFFIX" "$@"
tinydistill run "${common[@]}" --out "runs/banking77-k8$SUFFIX" --gold-per-class 8 --upper-bound "$@"

for name in "banking77$SUFFIX:results$SUFFIX" "banking77-k8$SUFFIX:results-k8$SUFFIX"; do
  cp "runs/${name%%:*}/results.json" "examples/banking77/${name##*:}.json"
  cp "runs/${name%%:*}/results.md" "examples/banking77/${name##*:}.md"
done
python examples/update_readme.py
