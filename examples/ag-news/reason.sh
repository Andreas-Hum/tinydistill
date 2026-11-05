#!/usr/bin/env bash
# Does a short reasoning step before the answer make the 1.5B teacher better? The same 3,000 training and 2,000
# test articles, scored directly and with --reason.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
data=hf:fancyzhx/ag_news
common=(examples/ag-news/task.yaml --data "$data:train" --limit 3000 --test "$data:test" --test-limit 2000
  --teacher Qwen/Qwen2.5-1.5B-Instruct --cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" --int8
  --seeds 3 --epochs 10 --teacher-devices cuda)
tinydistill run "${common[@]}" --out runs/ag-news-1.5b-plain3k
tinydistill run "${common[@]}" --reason --out runs/ag-news-1.5b-reason3k
for r in plain3k reason3k; do
  cp "runs/ag-news-1.5b-$r/results.json" "examples/ag-news/ag-news-1.5b-$r.json"
  cp "runs/ag-news-1.5b-$r/results.md" "examples/ag-news/ag-news-1.5b-$r.md"
done
tinydistill compare runs/ag-news-1.5b-plain3k runs/ag-news-1.5b-reason3k > examples/ag-news/reason.md
