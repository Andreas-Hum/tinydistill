#!/usr/bin/env bash
# CLINC150 with the `outside` label (examples/clinc150/outside.yaml): the dataset's out-of-scope requests, in the
# training and the test texts, are the label "none of these", listed last in the prompt, instead of the label
# "out of scope" among the intents. The 7B teacher (4-bit) scores the 20 most similar labels and always the
# outside label, as the third run of run.sh does with "out of scope":
#   - zero-shot, to set beside clinc150-7b-shortlist20-keep-oos
#   - few-shot: each text's prompt shows its 16 most similar of 8 gold examples per label (outside included)
# The results show in-scope accuracy and out-of-scope recall and precision next to the usual columns.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
data=hf:clinc/clinc_oos:plus
base=(examples/clinc150/outside.yaml --data "$data:train" --test "$data:test" --cache-dir runs/cache --device cuda
      --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --teacher Qwen/Qwen2.5-7B-Instruct --load-4bit
      --val-share 0.05 --teacher-devices cuda --shortlist 20 --shortlist-keep oos)
tinydistill run "${base[@]}" --out runs/clinc150-7b-outside
tinydistill run "${base[@]}" --few-shot 8 --few-shot-similar 16 --gold-per-class 8 \
  --out runs/clinc150-7b-outside-similar16
for run in clinc150-7b-outside clinc150-7b-outside-similar16; do
  cp "runs/$run/results.json" "examples/clinc150/$run.json"
  cp "runs/$run/results.md" "examples/clinc150/$run.md"
done
tinydistill compare runs/clinc150-7b-shortlist20-keep-oos runs/clinc150-7b-outside \
  runs/clinc150-7b-outside-similar16 > examples/clinc150/outside.md
