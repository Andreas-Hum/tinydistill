#!/usr/bin/env bash
# Reproduces the financial-tweets results in the README. The validation split is the test set. Labelling and
# training run on the GPU, student benchmarks on the CPU; every run has 3 training seeds and an upper bound.
#   - Qwen2.5-1.5B zero-shot and 8-shot (the 8 examples per class are also used in the student's loss, and a
#     gold-only baseline is trained on them)
#   - Qwen2.5-7B (4-bit), and an ensemble of three model families in 4 bit: Qwen2.5-7B, Mistral-7B and
#     Phi-3.5-mini (the Qwen member's scores come from the cache)
#   - a student trained only on synthetic texts that Qwen2.5-7B wrote from the task description
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
data=hf:zeroshot/twitter-financial-news-sentiment
task=examples/financial-tweets/task.yaml
base=("$task" --test "$data:validation" --cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS"
      --int8 --seeds 3 --epochs 5)  # 5 epochs: the default before 0.3.1
q7=Qwen/Qwen2.5-7B-Instruct

real=(--data "$data:train" --upper-bound)  # the synthetic run has no gold labels, so no upper bound there

tinydistill run "${base[@]}" "${real[@]}" --out runs/financial-tweets-1.5b --teacher-devices cpu,cuda
tinydistill run "${base[@]}" "${real[@]}" --few-shot 8 --gold-per-class 8 --no-probe --gold-only \
  --out runs/financial-tweets-1.5b-8shot --teacher-devices cpu,cuda
tinydistill run "${base[@]}" "${real[@]}" --teacher "$q7" --load-4bit --out runs/financial-tweets-7b \
  --teacher-devices cuda
# the same student trained for 20 epochs: does training longer help on this task too?
tinydistill run "${base[@]}" --data "$data:train" --teacher "$q7" --load-4bit --epochs 20 \
  --out runs/financial-tweets-7b-20-epochs --skip-teacher-bench
tinydistill run "${base[@]}" "${real[@]}" --load-4bit --teacher-devices cuda \
  --teacher "$q7,mistralai/Mistral-7B-Instruct-v0.3,microsoft/Phi-3.5-mini-instruct" \
  --out runs/financial-tweets-ensemble

tinydistill synth "$task" --per-label 500 --teacher "$q7" --load-4bit --device cuda --out runs/financial-tweets-synth.jsonl
tinydistill run "${base[@]}" --data runs/financial-tweets-synth.jsonl --teacher "$q7" --load-4bit \
  --out runs/financial-tweets-synthetic --skip-teacher-bench

for run in financial-tweets-1.5b financial-tweets-1.5b-8shot financial-tweets-7b financial-tweets-7b-20-epochs \
           financial-tweets-ensemble financial-tweets-synthetic; do
  cp "runs/$run/results.json" "examples/financial-tweets/$run.json"
  cp "runs/$run/results.md" "examples/financial-tweets/$run.md"
done
tinydistill compare runs/financial-tweets-1.5b runs/financial-tweets-1.5b-8shot runs/financial-tweets-7b \
  runs/financial-tweets-ensemble runs/financial-tweets-synthetic > examples/financial-tweets/teachers.md
python examples/update_readme.py
