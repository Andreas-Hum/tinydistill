#!/usr/bin/env bash
# Danish intents (MASSIVE da-DK, 60 intents): an English prompt over Danish texts. Labelling and training run
# on the GPU; student benchmarks on the CPU. Runs share one teacher cache.
#   - Qwen2.5 7B (4-bit), zero-shot and with 8 gold examples per class in the prompt (the 8-shot run also uses
#     those labels in the student's loss)
#   - students: the default English MiniLM-L6 and a multilingual MiniLM-L12, on the 8-shot teacher's labels
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
data=hf:mteb/amazon_massive_intent:da
base=(examples/massive-da/task.yaml --data "$data:train" --test "$data:test" --cache-dir runs/cache
      --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --teacher Qwen/Qwen2.5-7B-Instruct --load-4bit)
few=(--few-shot 8 --gold-per-class 8 --no-probe)

tinydistill run "${base[@]}" --out runs/massive-da-7b --teacher-devices cuda
tinydistill run "${base[@]}" "${few[@]}" --gold-only --upper-bound --out runs/massive-da-7b-8shot \
  --teacher-devices cuda
tinydistill run "${base[@]}" "${few[@]}" --student sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --out runs/massive-da-students/paraphrase-multilingual-MiniLM-L12-v2 --skip-teacher-bench

for run in massive-da-7b massive-da-7b-8shot massive-da-students/paraphrase-multilingual-MiniLM-L12-v2; do
  cp "runs/$run/results.json" "examples/massive-da/$(basename "$run").json"
  cp "runs/$run/results.md" "examples/massive-da/$(basename "$run").md"
done
tinydistill compare runs/massive-da-7b runs/massive-da-7b-8shot \
  runs/massive-da-students/paraphrase-multilingual-MiniLM-L12-v2 > examples/massive-da/summary.md
