#!/usr/bin/env bash
# DBpedia level 2 (70 article types, very unevenly common) from the FewMany benchmark, with the default for
# about 8 gold labels per class (docs/when-to-use.md) but the zero-shot prompt: Gemma 4 12B in 4 bit, the probe
# alone, and the student on the gold labels alone, for reference. With each text's 16 most similar gold
# examples in the prompt, labelling the long test articles would take about a day. FewMany reports 5-shot and 10-shot accuracy on the same test set (arXiv
# 2404.12365, Table 1). The teacher labels only the first 30,000 of the 240,942 training articles (--limit;
# the split is shuffled, and the rarest type has 11 among them), which also hold the gold labels. It
# still reads all 60,794 test articles, so labelling takes hours even zero-shot.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
out=examples/dbpedia70
data=hf:FastFit/dbpedia_70
common=(examples/dbpedia70/task.yaml --data "$data:train" --limit 30000 --test "$data:test"
  --cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --load-4bit
  --teacher google/gemma-4-12B-it --gold-per-class 8 --teacher-devices cuda)
runs=()
go() {  # name, then its own arguments
  local name=$1; shift
  tinydistill run "${common[@]}" "$@" --out "runs/$name"
  cp "runs/$name/results.json" "$out/$name.json"
  cp "runs/$name/results.md" "$out/$name.md"
  runs+=("runs/$name")
  tinydistill compare "${runs[@]}" > "$out/summary.md"
}
go dbpedia70-gemma4-12b-probe-only-zeroshot --probe-only --gold-only --few-shot-similar 0
