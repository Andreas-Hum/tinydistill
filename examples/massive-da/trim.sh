#!/usr/bin/env bash
# The multilingual student of run.sh with its vocabulary trimmed to the Danish training texts (train --trim-vocab):
# the same 8-shot teacher labels (from the score cache), the same settings, a smaller model.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
data=hf:mteb/amazon_massive_intent:da
out=runs/massive-da-students/paraphrase-multilingual-MiniLM-L12-v2-trimmed
tinydistill run examples/massive-da/task.yaml --data "$data:train" --test "$data:test" --cache-dir runs/cache \
  --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --teacher Qwen/Qwen2.5-7B-Instruct --load-4bit \
  --few-shot 8 --gold-per-class 8 --student sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --trim-vocab --out $out --skip-teacher-bench
cp $out/results.json examples/massive-da/paraphrase-multilingual-MiniLM-L12-v2-trimmed.json
cp $out/results.md examples/massive-da/paraphrase-multilingual-MiniLM-L12-v2-trimmed.md
tinydistill compare runs/massive-da-students/paraphrase-multilingual-MiniLM-L12-v2 runs/massive-da-students/paraphrase-multilingual-MiniLM-L12-v2-trimmed > examples/massive-da/trim.md
