#!/usr/bin/env bash
# The probe teacher (label --probe): a linear probe on the 7B teacher's hidden states, fit on the 8 gold examples
# per class, averaged with its in-context scores. Against the same teacher without the probe: on Banking77
# runs/banking77-7b-similar16, on CLINC150 runs/clinc150-7b-similar16 (from sets.sh). The scored texts are
# scored once more, to record their states.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
out=examples/teachers-2026/probe
mkdir -p "$out"
common=(--cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --load-4bit
  --teacher Qwen/Qwen2.5-7B-Instruct --few-shot 8 --few-shot-similar 16 --gold-per-class 8 --probe
  --teacher-devices cuda)
tinydistill run examples/banking77/task.yaml --data hf:PolyAI/banking77:train --test hf:PolyAI/banking77:test \
  --epochs 10 "${common[@]}" --out runs/banking77-7b-similar16-probe
tinydistill run examples/clinc150/task.yaml --data hf:clinc/clinc_oos:plus:train \
  --test hf:clinc/clinc_oos:plus:test --shortlist 20 --shortlist-keep oos --val-share 0.05 "${common[@]}" \
  --out runs/clinc150-7b-similar16-probe
for r in banking77-7b-similar16-probe clinc150-7b-similar16-probe; do
  cp "runs/$r/results.json" "$out/$r.json"; cp "runs/$r/results.md" "$out/$r.md"
done
tinydistill compare runs/banking77-7b-similar16 runs/banking77-7b-similar16-probe > "$out/banking77.md"
tinydistill compare runs/clinc150-7b-similar16 runs/clinc150-7b-similar16-probe > "$out/clinc150.md"
