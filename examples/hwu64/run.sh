#!/usr/bin/env bash
# HWU64 (64 intents), a task none of the settings were tuned on, with the settings of "Which settings to use"
# (docs/when-to-use.md) for about 8 gold labels per class: Gemma 4 12B in 4 bit, the probe on its hidden states.
#   1. the probe alone with a zero-shot prompt (minutes), and the student on the 512 gold labels alone
#   2. the probe alone with each text's 16 most similar gold examples
#   3. the teacher scoring every label with the same examples, averaged with the probe (hours)
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
out=examples/hwu64
data=hf:DeepPavlov/hwu64
common=(examples/hwu64/task.yaml --data "$data:train" --test "$data:test" --cache-dir runs/cache --device cuda
  --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --load-4bit --teacher google/gemma-4-12B-it --gold-per-class 8
  --teacher-devices cuda)
similar=(--few-shot 8 --few-shot-similar 16)
runs=()
go() {  # name, then its own arguments
  local name=$1; shift
  tinydistill run "${common[@]}" "$@" --out "runs/$name"
  cp "runs/$name/results.json" "$out/$name.json"
  cp "runs/$name/results.md" "$out/$name.md"
  runs+=("runs/$name")
  tinydistill compare "${runs[@]}" > "$out/summary.md"
}
go hwu64-gemma4-12b-probe-only-zeroshot --probe-only --gold-only --few-shot-similar 0
go hwu64-gemma4-12b-probe-only-similar16 "${similar[@]}" --probe-only
go hwu64-gemma4-12b-similar16 "${similar[@]}" --probe
