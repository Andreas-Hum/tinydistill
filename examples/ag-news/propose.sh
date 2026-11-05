#!/usr/bin/env bash
# The teacher rewrites the task's instruction (tune --propose, three rounds on its dev mistakes), then the best
# task file is used alone and together with the original as a prompt ensemble, with run.sh's settings.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
data=hf:fancyzhx/ag_news
teacher=Qwen/Qwen2.5-1.5B-Instruct
tinydistill tune examples/ag-news/task.yaml --propose 5 --rounds 3 --data "$data:train" --limit 3000 \
  --dev-size 200 --teacher "$teacher" --device cuda --out examples/ag-news/propose-1.5b
common=(--data "$data:train" --limit 10000 --test "$data:test" --teacher "$teacher" --cache-dir runs/cache
  --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --val-share 0.05 --teacher-devices cuda)
tinydistill run examples/ag-news/propose-1.5b-best.yaml "${common[@]}" --out runs/ag-news-1.5b-proposed
tinydistill run examples/ag-news/task.yaml --prompts examples/ag-news/propose-1.5b-best.yaml "${common[@]}" \
  --out runs/ag-news-1.5b-ensemble
for r in proposed ensemble; do
  cp "runs/ag-news-1.5b-$r/results.json" "examples/ag-news/ag-news-1.5b-$r.json"
  cp "runs/ag-news-1.5b-$r/results.md" "examples/ag-news/ag-news-1.5b-$r.md"
done
tinydistill compare runs/ag-news-1.5b runs/ag-news-1.5b-proposed runs/ag-news-1.5b-ensemble > examples/ag-news/propose-runs.md
