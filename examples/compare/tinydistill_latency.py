"""Time a saved tinydistill student with the same protocol as the other methods (tinydistill's Predictor, the
ONNX int8 model, batch 1, 2 threads, the first 100 test texts), so every latency in the tables comes from the
same machine and the same texts. Takes the run's saved results file and its run directory (with onnx/ and
test.jsonl, in the test set's order). Writes examples/compare/results/tinydistill-<results file name>.json.

    python examples/compare/tinydistill_latency.py examples/banking77/banking77-7b-similar16.json \
        runs/banking77-7b-similar16
"""

import argparse
import json
from pathlib import Path

import common

from tinydistill.export import Predictor

p = argparse.ArgumentParser()
p.add_argument("results", type=Path, help="the saved results.json copy in examples/")
p.add_argument("run", type=Path, help="its run directory")
a = p.parse_args()

saved = json.loads(a.results.read_text(encoding="utf-8"))
test = [json.loads(line) for line in (a.run / "test.jsonl").read_text(encoding="utf-8").splitlines() if line]
test_texts, test_gold = [r["text"] for r in test], [r["gold"] for r in test]
pr = Predictor(a.run / "onnx", int8=True, threads=common.LATENCY_THREADS)
pred = pr.logits(test_texts).argmax(-1).tolist()
common.write(
    f"tinydistill-{a.results.stem}",
    {
        "task": saved["task"],
        "method": "tinydistill",
        "results_file": a.results.resolve().relative_to(common.EXAMPLES.resolve()).as_posix(),
        "model": saved["student"],
        "n_test": len(test_texts),
        **common.scores(pred, test_gold),  # should match the saved ONNX int8 row
        "latency_ms_b1": common.latency_ms(lambda b: pr.logits(b, batch_size=1), test_texts),
        "latency_threads": common.LATENCY_THREADS,
    },
)
