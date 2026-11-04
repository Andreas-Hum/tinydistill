#!/usr/bin/env bash
# Long texts: 5,000 IMDB training reviews and 2,000 test reviews (shuffled, seed 0), Qwen2.5-1.5B as
# the teacher. The same student either reads the first 128 tokens of a review, or reads every review in
# windows of 128 tokens, 64 apart, and averages them.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
# the splits are sorted by label, so take shuffled subsets first
python - <<'PY'
import json
from pathlib import Path

from datasets import load_dataset

Path("runs/data").mkdir(parents=True, exist_ok=True)
for split, n in (("train", 5000), ("test", 2000)):
    ds = load_dataset("stanfordnlp/imdb", split=split).shuffle(seed=0).select(range(n))
    with open(f"runs/data/imdb-{split}.jsonl", "w", encoding="utf-8") as f:
        for r in ds:
            f.write(json.dumps({"text": r["text"], "label": ["neg", "pos"][r["label"]]}) + "\n")
PY
base=(examples/imdb/task.yaml --data runs/data/imdb-train.jsonl --test runs/data/imdb-test.jsonl --cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3
      --teacher Qwen/Qwen2.5-1.5B-Instruct --teacher-devices cuda)
tinydistill run "${base[@]}" --out runs/imdb-truncated
tinydistill run "${base[@]}" --chunk-tokens 128 --chunk-stride 64 --out runs/imdb-chunked
for run in imdb-truncated imdb-chunked; do
  cp "runs/$run/results.json" "examples/imdb/$run.json"
  cp "runs/$run/results.md" "examples/imdb/$run.md"
done
tinydistill compare runs/imdb-truncated runs/imdb-chunked > examples/imdb/summary.md
