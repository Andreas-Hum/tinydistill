#!/usr/bin/env bash
# Claim Stance (55 debate topics) from the FewMany benchmark, with the default for about 8 gold labels per
# class (docs/when-to-use.md): Gemma 4 12B in 4 bit, the probe alone, first with the zero-shot prompt (and the
# student on the gold labels alone, for reference), then with each text's 16 most similar gold examples in the
# prompt. FewMany reports 5-shot and 10-shot accuracy on the same test set (arXiv 2404.12365, Table 1). The
# topics with fewer than 8 training claims give all they have.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
out=examples/claim-stance55
data=hf:FastFit/claim_stance_55
common=(examples/claim-stance55/task.yaml --data "$data:train" --test "$data:test"
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
go claim-stance55-gemma4-12b-probe-only-zeroshot --probe-only --gold-only --few-shot-similar 0
go claim-stance55-gemma4-12b-probe-only-similar16 --few-shot 8 --few-shot-similar 16
