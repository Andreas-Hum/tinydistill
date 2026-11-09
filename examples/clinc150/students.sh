#!/usr/bin/env bash
# Larger students on the labels of the best CLINC150 teacher (examples/teachers-2026/probe-gemma.sh: Gemma 4 12B
# with 16 similar examples and the probe, read from the score cache). CLINC150 is the task where that teacher
# is furthest above its MiniLM-L6 student (0.941 against 0.860), so it is where a larger student could gain.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
out=examples/clinc150/students
mkdir -p "$out"
common=(examples/clinc150/task.yaml --data hf:clinc/clinc_oos:plus:train --test hf:clinc/clinc_oos:plus:test
  --shortlist 20 --shortlist-keep oos --val-share 0.05 --cache-dir runs/cache --device cuda
  --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --load-4bit --teacher google/gemma-4-12B-it --few-shot 8
  --few-shot-similar 16 --gold-per-class 8 --probe --teacher-devices cuda --skip-teacher-bench)
for student in sentence-transformers/all-MiniLM-L12-v2 sentence-transformers/all-mpnet-base-v2; do
  run=runs/clinc150-students/${student##*/}
  tinydistill run "${common[@]}" --student "$student" --out "$run" "$@"
  [[ " $* " == *" --dry-run "* ]] && continue
  cp "$run/results.json" "$out/${student##*/}.json"
  cp "$run/results.md" "$out/${student##*/}.md"
done
[[ " $* " == *" --dry-run "* ]] && exit 0
tinydistill compare runs/clinc150-gemma4-12b-similar16-probe runs/clinc150-students/* > "$out/summary.md"
