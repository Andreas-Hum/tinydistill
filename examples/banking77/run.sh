#!/usr/bin/env bash
# Reproduces the Banking77 results in the README. Labelling and training run on the GPU; every student
# benchmark runs on the CPU. Runs share one teacher cache, so each text is labelled once per teacher prompt.
#   - four teachers: Qwen2.5 1.5B and 7B (4-bit), each zero-shot and with 8 gold examples per class in the
#     prompt; the 8-shot runs also use those labels in the student's loss and train a gold-only baseline
#   - every run: 3 training seeds and an upper bound (the student on all gold labels)
#   - student sizes, trained on the 7B 8-shot teacher's labels, and a static-embedding student
# The 1.5B zero-shot teacher is also timed on the CPU; the others only on the GPU (4-bit loading needs CUDA,
# and the 8-shot prompts are long).
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
base=(examples/banking77/task.yaml --data hf:PolyAI/banking77:train --test hf:PolyAI/banking77:test
      --cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3
      --epochs 5)  # the default before 0.3.1; the lines that pass --epochs 20 override it
few=(--few-shot 8 --gold-per-class 8 --gold-only)
big=(--teacher Qwen/Qwen2.5-7B-Instruct --load-4bit)

tinydistill run "${base[@]}" --upper-bound --out runs/banking77-1.5b --teacher-devices cpu,cuda
tinydistill run "${base[@]}" --upper-bound "${few[@]}" --out runs/banking77-1.5b-8shot --teacher-devices cuda
tinydistill run "${base[@]}" --upper-bound "${big[@]}" --out runs/banking77-7b --teacher-devices cuda
tinydistill run "${base[@]}" --upper-bound "${big[@]}" "${few[@]}" --out runs/banking77-7b-8shot \
  --teacher-devices cuda
for student in sentence-transformers/all-MiniLM-L12-v2 BAAI/bge-small-en-v1.5 \
               sentence-transformers/all-mpnet-base-v2; do
  tinydistill run "${base[@]}" "${big[@]}" --few-shot 8 --gold-per-class 8 --student "$student" \
    --out "runs/banking77-students/${student##*/}" --skip-teacher-bench
done
# a static-embedding student (no attention), which wants a higher learning rate and more epochs
tinydistill run "${base[@]}" "${big[@]}" --few-shot 8 --gold-per-class 8 --student static --lr 1e-3 --epochs 20 \
  --out runs/banking77-students/static --skip-teacher-bench
# MiniLM-L6 at 10 epochs (the default from 0.3.1), and the two smallest transformer students trained as long
# as the static one, for a like-for-like comparison
tinydistill run "${base[@]}" "${big[@]}" --few-shot 8 --gold-per-class 8 --epochs 10 \
  --out runs/banking77-students/all-MiniLM-L6-v2-10-epochs --skip-teacher-bench
for student in sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/all-MiniLM-L12-v2; do
  tinydistill run "${base[@]}" "${big[@]}" --few-shot 8 --gold-per-class 8 --student "$student" --epochs 20 \
    --out "runs/banking77-students/${student##*/}-20-epochs" --skip-teacher-bench
done

for run in banking77-1.5b banking77-1.5b-8shot banking77-7b banking77-7b-8shot; do
  cp "runs/$run/results.json" "examples/banking77/$run.json"
  cp "runs/$run/results.md" "examples/banking77/$run.md"
done
tinydistill compare runs/banking77-1.5b runs/banking77-1.5b-8shot runs/banking77-7b runs/banking77-7b-8shot \
  > examples/banking77/teachers.md
mkdir -p examples/banking77/students
for dir in runs/banking77-students/*/; do cp "$dir/results.json" "examples/banking77/students/$(basename "$dir").json"; done
tinydistill compare runs/banking77-7b-8shot runs/banking77-students/* > examples/banking77/students.md
python - <<'PY'  # the charts in the README
from tinydistill import plots
plots.speed_accuracy("runs/banking77-7b-8shot").savefig("docs/images/banking77-speed-accuracy.png", dpi=130)
plots.soft_labels("runs/banking77-students/all-MiniLM-L6-v2-20-epochs").savefig(
    "docs/images/banking77-soft-labels.png", dpi=130
)
PY
python examples/update_readme.py
