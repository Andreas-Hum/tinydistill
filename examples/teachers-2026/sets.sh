#!/usr/bin/env bash
# Demonstration ensembling (label --few-shot-sets 3): each text is scored with three buckets of its similar examples
# (ranks 1-16, 17-32, 33-48) and the label probabilities are averaged, against one bucket of 16. The 7B teacher of
# banking77/similar.sh; on Banking77 the one-bucket run is runs/banking77-7b-similar16.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
out=examples/teachers-2026/sets
mkdir -p "$out"
common=(--cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --load-4bit
  --teacher Qwen/Qwen2.5-7B-Instruct --few-shot 8 --few-shot-similar 16 --gold-per-class 8 --no-probe --teacher-devices cuda)
tinydistill run examples/banking77/task.yaml --data hf:PolyAI/banking77:train --test hf:PolyAI/banking77:test \
  --epochs 10 "${common[@]}" --few-shot-sets 3 --out runs/banking77-7b-similar16-sets3
clinc=(examples/clinc150/task.yaml --data hf:clinc/clinc_oos:plus:train --test hf:clinc/clinc_oos:plus:test
  --shortlist 20 --shortlist-keep oos --val-share 0.05)
tinydistill run "${clinc[@]}" "${common[@]}" --out runs/clinc150-7b-similar16
tinydistill run "${clinc[@]}" "${common[@]}" --few-shot-sets 3 --out runs/clinc150-7b-similar16-sets3
for r in banking77-7b-similar16-sets3 clinc150-7b-similar16 clinc150-7b-similar16-sets3; do
  cp "runs/$r/results.json" "$out/$r.json"; cp "runs/$r/results.md" "$out/$r.md"
done
tinydistill compare runs/banking77-7b-similar16 runs/banking77-7b-similar16-sets3 > "$out/banking77.md"
tinydistill compare runs/clinc150-7b-similar16 runs/clinc150-7b-similar16-sets3 > "$out/clinc150.md"
