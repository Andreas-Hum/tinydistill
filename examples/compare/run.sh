#!/usr/bin/env bash
# How a tinydistill student compares with what people use instead, on Banking77, CLINC150 (plus), Danish
# MASSIVE and financial tweets, with the same 8 gold labels per class and the same test set. Everything here
# runs on the CPU; the tinydistill and LLM rows are read from the saved GPU runs in examples/ (see each task's
# run.sh), and re-timed here when their run directories are in runs/.
#   - gold only: the student's encoder fine-tuned on the gold labels alone (gold_only.py, tinydistill's own
#     training function)
#   - SetFit with MiniLM-L6 and all-mpnet-base-v2 (and the multilingual MiniLM-L12 on Danish MASSIVE), 3 seeds
#   - GLiClass zero-shot, base and large v3.0, with the label names (and descriptions where the task has them)
# Each runner writes examples/compare/results/<method>-<task>-<model>.json; table.py writes
# examples/compare/<task>.md from those and the saved runs, and update_readme.py puts the tables into
# docs/how-it-compares.md. Rerun one runner on its own to add or refresh a row, then the last two commands.
#
# SetFit and GLiClass run in their own environment, kept out of tinydistill's dependencies:
#   python -m venv compare-venv
#   compare-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
#   compare-venv/bin/pip install setfit gliclass
#   COMPARE_PYTHON=compare-venv/bin/python bash examples/compare/run.sh
# Latency is timed with 2 threads whatever OMP_NUM_THREADS says; --threads sets the threads for training.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
cmp=${COMPARE_PYTHON:-python}
export PYTHONPATH=$PWD/src${PYTHONPATH:+:$PYTHONPATH}  # the runners import tinydistill's data and metrics
threads=16

multi=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# saved results file : run directory, for every tinydistill row the tables can pick
for pair in banking77/banking77-7b-similar16:banking77-7b-similar16 \
            banking77/banking77-qwen35-9b-similar16:banking77-qwen35-9b-similar16 \
            teachers-2026/probe/banking77-7b-similar16-probe:banking77-7b-similar16-probe \
            clinc150/clinc150-7b-shortlist20-keep-oos:clinc150-7b-shortlist20-keep-oos \
            clinc150/clinc150-7b-outside-similar16:clinc150-7b-outside-similar16 \
            teachers-2026/probe/clinc150-7b-similar16-probe:clinc150-7b-similar16-probe \
            teachers-2026/similar/clinc150-gemma4-12b:clinc150-gemma4-12b-similar16 \
            teachers-2026/probe/clinc150-gemma4-12b-similar16-probe:clinc150-gemma4-12b-similar16-probe \
            massive-da/paraphrase-multilingual-MiniLM-L12-v2:massive-da-students/${multi##*/} \
            massive-da/${multi##*/}-trimmed:massive-da-students/${multi##*/}-trimmed \
            teachers-2026/similar/massive-da-ensemble:massive-da-ensemble-similar16 \
            massive-da/${multi##*/}-trimmed-ensemble-similar16:massive-da-students/${multi##*/}-trimmed-ensemble-similar16 \
            financial-tweets/financial-tweets-ensemble:financial-tweets-ensemble; do
  saved=${pair%%:*} run=${pair#*:}
  if [ -d "runs/$run/onnx" ]; then
    python examples/compare/tinydistill_latency.py "examples/$saved.json" "runs/$run"
  fi
done

# the gold-only baseline trains as long as the tinydistill student it is compared with
python examples/compare/gold_only.py banking77 clinc150 --epochs 10 --threads $threads
python examples/compare/gold_only.py massive-da --epochs 10 --threads $threads \
  --student "$multi"
python examples/compare/gold_only.py financial-tweets --epochs 5 --threads $threads

for task in banking77 clinc150 massive-da financial-tweets; do
  "$cmp" examples/compare/setfit_runner.py "$task" --body MiniLM-L6 --body mpnet-base --threads $threads
done
"$cmp" examples/compare/setfit_runner.py massive-da --body multilingual-MiniLM-L12 --threads $threads

for size in base large; do
  "$cmp" examples/compare/gliclass_runner.py banking77 clinc150 massive-da financial-tweets --size $size \
    --threads $threads
  "$cmp" examples/compare/gliclass_runner.py clinc150 financial-tweets --size $size --descriptions \
    --threads $threads
done

python examples/compare/table.py
python examples/update_readme.py
