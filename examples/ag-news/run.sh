#!/usr/bin/env bash
# AG News with the 1.5B teacher, as in the "your own task" notebook: the first 10,000 training articles, the full
# test set (7,600), 3 training seeds, best-epoch selection and an upper bound.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
data=hf:fancyzhx/ag_news
tinydistill run examples/ag-news/task.yaml --data "$data:train" --limit 10000 --test "$data:test" \
  --cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --upper-bound \
  --val-share 0.05 --out runs/ag-news-1.5b --teacher-devices cuda
cp runs/ag-news-1.5b/results.json examples/ag-news/ag-news-1.5b.json
cp runs/ag-news-1.5b/results.md examples/ag-news/ag-news-1.5b.md
python examples/update_readme.py
