#!/usr/bin/env bash
# Before a release: label and train one fixed benchmark from scratch and compare it with its saved reference,
# as llama.cpp checks perplexity and speed. Banking77 with Qwen2.5-7B (4 bit), the probe alone on a zero-shot
# prompt (examples/teachers-2026/probe-only.sh): a few minutes on a GPU. A fresh score cache, so labelling is
# timed; --few-shot-similar 0 keeps the reference's zero-shot prompt. Fails when the deployed student is more
# than a point less accurate or labelling takes 1.5 times as long.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
run=runs/benchmark-check
rm -rf "$run" "$run-cache"
tinydistill run examples/banking77/task.yaml --data hf:PolyAI/banking77:train --test hf:PolyAI/banking77:test \
  --epochs 10 --cache-dir "$run-cache" --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --load-4bit \
  --teacher Qwen/Qwen2.5-7B-Instruct --gold-per-class 8 --probe-only --few-shot-similar 0 --teacher-devices cuda \
  --out "$run"
python examples/benchmark/check.py examples/teachers-2026/probe-only/banking77-7b-probe-only-zeroshot.json \
  "$run/results.json"
