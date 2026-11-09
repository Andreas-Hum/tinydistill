#!/usr/bin/env bash
# An mpnet-base student (110M parameters, against MiniLM-L6's 23M) on the labels of each task's best saved run,
# with `train --out`: the teacher is not run again. On CLINC150 (examples/clinc150/students.sh) it gained 4
# points; here Banking77, HWU64 and Danish MASSIVE (the multilingual mpnet, trimmed as the run's student was).
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
out=examples/students
common=(--epochs 10 --gold-per-class 8 --seeds 3 --device cuda --threads "$OMP_NUM_THREADS")

student() {  # the run with the labels, the new run's name, then train's own arguments
  local run=$1 name=$2; shift 2
  tinydistill train "runs/$run" "${common[@]}" "$@" --out "runs/students/$name"
  tinydistill export "runs/students/$name" --int8
  tinydistill eval "runs/students/$name" --threads "$OMP_NUM_THREADS" --skip-teacher-bench
  cp "runs/students/$name/results.json" "$out/$name.json"
  cp "runs/students/$name/results.md" "$out/$name.md"
}

student banking77-7b-similar16-probe banking77-mpnet --student sentence-transformers/all-mpnet-base-v2
student hwu64-gemma4-12b-similar16 hwu64-mpnet --student sentence-transformers/all-mpnet-base-v2
student massive-da-students/paraphrase-multilingual-MiniLM-L12-v2-trimmed-gemma4-12b-similar16-probe \
  massive-da-multilingual-mpnet --student sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --trim-vocab
tinydistill compare runs/banking77-7b-similar16-probe runs/students/banking77-mpnet runs/hwu64-gemma4-12b-similar16 \
  runs/students/hwu64-mpnet \
  runs/massive-da-students/paraphrase-multilingual-MiniLM-L12-v2-trimmed-gemma4-12b-similar16-probe \
  runs/students/massive-da-multilingual-mpnet > "$out/mpnet.md"
