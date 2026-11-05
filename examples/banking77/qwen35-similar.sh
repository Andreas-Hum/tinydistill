#!/usr/bin/env bash
# similar.sh with Qwen3.5-9B as the teacher: each text's 16 most similar of the same 616 gold examples. Qwen3.5
# cannot pack texts (linear-attention layers), so it scores them as rows of a batch; the teacher's time per text
# is in the results.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
out=runs/banking77-qwen35-9b-similar16
tinydistill run examples/banking77/task.yaml --data hf:PolyAI/banking77:train --test hf:PolyAI/banking77:test \
  --cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --epochs 10 \
  --teacher Qwen/Qwen3.5-9B --load-4bit --few-shot 8 --few-shot-similar 16 --gold-per-class 8 \
  --out $out --teacher-devices cuda
cp $out/results.json examples/banking77/banking77-qwen35-9b-similar16.json
cp $out/results.md examples/banking77/banking77-qwen35-9b-similar16.md
