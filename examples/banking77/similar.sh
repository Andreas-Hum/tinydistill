#!/usr/bin/env bash
# Similar few-shot examples on Banking77: the 7B teacher shows each text the 16 most similar of the same 8 gold
# examples per class the 8-shot run shows all of (so both runs use the same gold labels).
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
tinydistill run examples/banking77/task.yaml --data hf:PolyAI/banking77:train --test hf:PolyAI/banking77:test \
  --cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --epochs 10 \
  --teacher Qwen/Qwen2.5-7B-Instruct --load-4bit --few-shot 8 --few-shot-similar 16 --gold-per-class 8 --no-probe \
  --out runs/banking77-7b-similar16 --teacher-devices cuda
cp runs/banking77-7b-similar16/results.json examples/banking77/banking77-7b-similar16.json
cp runs/banking77-7b-similar16/results.md examples/banking77/banking77-7b-similar16.md
