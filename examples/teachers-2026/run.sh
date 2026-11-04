#!/usr/bin/env bash
# Newer teachers (Qwen3.5, Gemma 4) against Qwen2.5-7B, all in 4 bit, on the dev splits `tune` uses: teacher
# accuracy only, before any student is trained. AG News uses the same 200 dev texts as examples/ag-news.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
teachers=Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B,google/gemma-4-E4B-it,google/gemma-4-12B-it
common=(--teacher "$teachers" --load-4bit --device cuda)
tinydistill tune examples/ag-news/task.yaml --data hf:fancyzhx/ag_news:train --limit 3000 --dev-size 200 \
  "${common[@]}" --out examples/teachers-2026/ag-news
tinydistill tune examples/banking77/task.yaml --data hf:PolyAI/banking77:train --dev-size 500 --few-shot 0,8 \
  "${common[@]}" --out examples/teachers-2026/banking77
tinydistill tune examples/massive-da/task.yaml --data hf:mteb/amazon_massive_intent:da:train --dev-size 500 \
  --few-shot 0,8 "${common[@]}" --out examples/teachers-2026/massive-da
