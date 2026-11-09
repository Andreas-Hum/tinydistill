"""int8 with one scale per weight matrix (tinydistill up to 0.7.0) against one scale per output channel (from
0.7.1): test accuracy and CPU latency (batch 1, 2 threads) of each, and of fp32, for exported runs. Both int8
models are made here from the run's fp32 ONNX model, so the comparison does not depend on the version that
exported the run.

    OMP_NUM_THREADS=2 python examples/export/per_channel.py runs/banking77-7b-similar16-probe \\
        runs/hwu64-gemma4-12b-similar16 runs/clinc150-gemma4-12b-similar16-probe \\
        runs/massive-da-students/paraphrase-multilingual-MiniLM-L12-v2-trimmed-gemma4-12b-similar16-probe \\
        runs/financial-tweets-7b runs/students/banking77-mpnet-pc > examples/export/per-channel.json

runs/students/banking77-mpnet-pc is examples/students/mpnet.sh's Banking77 student, trained again.
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from onnxruntime.quantization import QuantType, quantize_dynamic

from tinydistill.export import Predictor


def measure(folder: Path, int8: bool, texts: list[str], gold: np.ndarray) -> dict:
    p = Predictor(folder, int8=int8, threads=2)
    accuracy = float((p.logits(texts).argmax(-1) == gold).mean())
    for t in texts[:20]:  # warm up
        p.logits([t])
    t0 = time.perf_counter()
    for t in texts[:200]:
        p.logits([t])
    return {"accuracy": accuracy, "latency_ms_b1": (time.perf_counter() - t0) / 200 * 1000}


def compare(run: Path, tmp: Path) -> dict:
    rows = [json.loads(line) for line in (run / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    texts, gold = [r["text"] for r in rows], np.array([r["gold"] for r in rows])
    out: dict = {"run": run.name, "n_test": len(rows), "fp32": measure(run / "onnx", False, texts, gold)}
    for name, per_channel in [("int8_per_matrix", False), ("int8_per_channel", True)]:
        d = tmp / name
        shutil.copytree(run / "onnx", d)
        q = str(d / "model.int8.onnx")
        quantize_dynamic(str(d / "model.onnx"), q, weight_type=QuantType.QInt8, per_channel=per_channel)
        out[name] = measure(d, True, texts, gold)
    return out


if __name__ == "__main__":
    results = []
    for arg in sys.argv[1:]:
        with tempfile.TemporaryDirectory() as tmp:
            results.append(compare(Path(arg), Path(tmp)))
    print(json.dumps(results, indent=1))
