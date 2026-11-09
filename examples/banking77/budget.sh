#!/usr/bin/env bash
# Budgeted labelling on Banking77: the teacher of similar.sh labels only N of the 10,003 training texts, chosen at
# random (--rounds 1) or by the student's uncertainty over 4 rounds. The scores come from the score cache that
# similar.sh filled, so this measures the choice of texts, not new labelling.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
for n in 1000 2500 5000; do
  for r in 1 4; do
    out=runs/banking77-7b-similar16-budget$n-r$r
    tinydistill run examples/banking77/task.yaml --data hf:PolyAI/banking77:train --test hf:PolyAI/banking77:test \
      --cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --epochs 10 \
      --teacher Qwen/Qwen2.5-7B-Instruct --load-4bit --few-shot 8 --few-shot-similar 16 --gold-per-class 8 --no-probe \
      --budget $n --rounds $r --out $out --teacher-devices cuda
    cp $out/results.json examples/banking77/budget$n-r$r.json
    cp $out/results.md examples/banking77/budget$n-r$r.md
  done
done
python examples/banking77/budget_table.py
