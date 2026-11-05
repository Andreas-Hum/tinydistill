#!/usr/bin/env bash
# similar.sh with a SetFit-style contrastive warm-up of the student (--warmup-pairs 20000) before distillation.
# The teacher's labels come from the score cache of similar.sh, so only the student's training differs; compare
# with examples/banking77/banking77-7b-similar16.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
out=runs/banking77-7b-similar16-warmup
tinydistill run examples/banking77/task.yaml --data hf:PolyAI/banking77:train --test hf:PolyAI/banking77:test \
  --cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --epochs 10 \
  --teacher Qwen/Qwen2.5-7B-Instruct --load-4bit --few-shot 8 --few-shot-similar 16 --gold-per-class 8 \
  --warmup-pairs 20000 --skip-teacher-bench --out $out --teacher-devices cuda
cp $out/results.json examples/banking77/banking77-7b-similar16-warmup.json
cp $out/results.md examples/banking77/banking77-7b-similar16-warmup.md
# The same on a labelling budget, against budget.sh's random choice (--rounds 1) of 1,000 and 2,500 texts
for n in 1000 2500; do
  out=runs/banking77-7b-similar16-budget$n-warmup
  tinydistill run examples/banking77/task.yaml --data hf:PolyAI/banking77:train --test hf:PolyAI/banking77:test \
    --cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --epochs 10 \
    --teacher Qwen/Qwen2.5-7B-Instruct --load-4bit --few-shot 8 --few-shot-similar 16 --gold-per-class 8 \
    --budget $n --rounds 1 --warmup-pairs 20000 --skip-teacher-bench --out $out --teacher-devices cuda
  cp $out/results.json examples/banking77/budget$n-warmup.json
  cp $out/results.md examples/banking77/budget$n-warmup.md
done
python examples/banking77/warmup_table.py
