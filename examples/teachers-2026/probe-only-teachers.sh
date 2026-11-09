#!/usr/bin/env bash
# Two teachers with the default for about 8 gold labels per class (the probe alone, each text's 16 most similar
# of the gold examples in the prompt): Gemma 4 12B and Qwen3.5-9B, both in 4 bit. Gemma 4 12B's runs on
# CLINC150, Danish MASSIVE and HWU64 come from probe-only.sh and hwu64/run.sh; Banking77's is made here.
# Gemma 4 31B does not fit next to its activations on a 24 GB GPU (0.18 texts/s), so it is not compared.
# Pass task names to run only those.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
out=examples/teachers-2026/probe-only-teachers
mkdir -p "$out"
common=(--cache-dir runs/cache --device cuda --threads "$OMP_NUM_THREADS" --int8 --seeds 3 --load-4bit
  --gold-per-class 8 --probe-only --few-shot 8 --few-shot-similar 16 --teacher-devices cuda)
tasks=" ${*:-banking77 clinc150 massive-da hwu64} "

run() {  # name, tag, teacher, then the task's own arguments
  local name=$1 tag=$2 teacher=$3; shift 3
  [[ $tasks == *" $name "* ]] || return 0
  local run=$name-$tag-probe-only-similar16
  tinydistill run "$@" "${common[@]}" --teacher "$teacher" --out "runs/$run"
  cp "runs/$run/results.json" "$out/$run.json"
  cp "runs/$run/results.md" "$out/$run.md"
}

banking77=(examples/banking77/task.yaml --data hf:PolyAI/banking77:train --test hf:PolyAI/banking77:test --epochs 10)
clinc150=(examples/clinc150/task.yaml --data hf:clinc/clinc_oos:plus:train --test hf:clinc/clinc_oos:plus:test
  --shortlist 20 --shortlist-keep oos --val-share 0.05)
massive_da=(examples/massive-da/task.yaml --data hf:mteb/amazon_massive_intent:da:train
  --test hf:mteb/amazon_massive_intent:da:test --student sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  --trim-vocab)
hwu64=(examples/hwu64/task.yaml --data hf:DeepPavlov/hwu64:train --test hf:DeepPavlov/hwu64:test)

run banking77 gemma4-12b google/gemma-4-12B-it "${banking77[@]}"
for t in banking77 clinc150 massive_da hwu64; do
  declare -n args=$t
  run "${t/_/-}" qwen35-9b Qwen/Qwen3.5-9B "${args[@]}"
done
compare=()
for t in banking77 clinc150 massive-da hwu64; do
  for tag in gemma4-12b qwen35-9b; do compare+=("runs/$t-$tag-probe-only-similar16"); done
done
tinydistill compare "${compare[@]}" > "$out/summary.md"
