#!/usr/bin/env bash
# CLINC150: 151 labels. The 7B teacher (4-bit) labels the training texts once with full scoring and once with
# --shortlist 20 (a sentence-embedding model picks each text's 20 most similar labels; the teacher scores only
# those). The results compare the teachers' accuracy and labelling time, and the students trained on each.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
data=hf:clinc/clinc_oos:plus
base=(examples/clinc150/task.yaml --data "$data:train" --test "$data:test" --cache-dir runs/cache --device cuda
      --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --teacher Qwen/Qwen2.5-7B-Instruct --load-4bit
      --val-share 0.05 --teacher-devices cuda)
tinydistill run "${base[@]}" --shortlist 20 --out runs/clinc150-7b-shortlist20
tinydistill run "${base[@]}" --upper-bound --out runs/clinc150-7b-full
# the catch-all label "out of scope" resembles no particular request, so it is scored for every text as well
tinydistill run "${base[@]}" --shortlist 20 --shortlist-keep oos --out runs/clinc150-7b-shortlist20-keep-oos
for run in clinc150-7b-shortlist20 clinc150-7b-full clinc150-7b-shortlist20-keep-oos; do
  cp "runs/$run/results.json" "examples/clinc150/$run.json"
  cp "runs/$run/results.md" "examples/clinc150/$run.md"
done
tinydistill compare runs/clinc150-7b-full runs/clinc150-7b-shortlist20 runs/clinc150-7b-shortlist20-keep-oos \
  > examples/clinc150/teachers.md
python examples/update_readme.py
