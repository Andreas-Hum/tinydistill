#!/usr/bin/env bash
# Rounds of teacher-written texts for the label pairs the similar.sh student confuses most (tinydistill improve).
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
tinydistill improve runs/banking77-7b-similar16 --rounds 3 --per-round 300 --device cuda \
  --threads "$OMP_NUM_THREADS" --int8 --out runs/banking77-7b-similar16-improve
cp runs/banking77-7b-similar16-improve/improve.json examples/banking77/improve.json
cp runs/banking77-7b-similar16-improve/improve.md examples/banking77/improve.md
