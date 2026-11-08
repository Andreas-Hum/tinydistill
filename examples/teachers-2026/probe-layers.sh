#!/usr/bin/env bash
# Which layer should the probe read? label --probe-only with a fixed zero-shot prompt (a minute or a few per
# layer), the probe fit on 8 gold examples per class, reading the input of the LM head (the default) or the output
# of a layer nearer the middle. Writes examples/teachers-2026/probe-layers/<task>.{json,md}. Pass task names to run
# only those.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=examples/teachers-2026/probe-layers
mkdir -p "$out"
common=(--cache-dir runs/cache --device cuda --load-4bit --gold-per-class 8 --probe-only)
tasks=" ${*:-banking77 clinc150 massive-da} "

sweep() {  # name, teacher, layers, then the task's own arguments
  local name=$1 teacher=$2 layers=$3; shift 3
  [[ $tasks == *" $name "* ]] || return 0
  local runs=()
  for layer in default $layers; do
    local run=runs/probe-layers/$name-$layer extra=()
    [[ $layer == default ]] || extra=(--probe-layer "$layer")
    tinydistill label "$@" "${common[@]}" --teacher "$teacher" "${extra[@]}" --out "$run"
    runs+=("$run")
  done
  python examples/teachers-2026/probe_layers.py "$out/$name" "${runs[@]}"
}

sweep banking77 Qwen/Qwen2.5-7B-Instruct "-4 -7 -10 -14 -18" examples/banking77/task.yaml \
  --data hf:PolyAI/banking77:train --test hf:PolyAI/banking77:test
sweep clinc150 google/gemma-4-12B-it "-6 -12 -18 -24 -30" examples/clinc150/task.yaml \
  --data hf:clinc/clinc_oos:plus:train --test hf:clinc/clinc_oos:plus:test
sweep massive-da google/gemma-4-12B-it "-6 -12 -18 -24 -30" examples/massive-da/task.yaml \
  --data hf:mteb/amazon_massive_intent:da:train --test hf:mteb/amazon_massive_intent:da:test
