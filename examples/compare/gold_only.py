"""tinydistill's gold-only baseline on the CPU: the student encoder fine-tuned on the 8 gold labels per class
alone, no teacher, with the same training function, settings and seeds as `tinydistill train --gold-only`.
That command trains the distilled student first (on every training text); this runs only the baseline. The
first seed's model is exported to ONNX int8 and timed with tinydistill's Predictor. Writes
examples/compare/results/gold-only-<task>-<student>.json.

    python examples/compare/gold_only.py banking77 --epochs 10 --threads 16
"""

import argparse
import tempfile
import time
from pathlib import Path

import common
import torch

from tinydistill.evaluate import dir_size_mb
from tinydistill.export import Predictor, export_onnx
from tinydistill.student import DEFAULT_STUDENT, load_student, student_logits, train_student

p = argparse.ArgumentParser()
p.add_argument("tasks", nargs="+", choices=list(common.TASKS))
p.add_argument("--student", default=DEFAULT_STUDENT)
p.add_argument("--epochs", type=int, default=10, help="as the tinydistill student it is compared with")
p.add_argument("--threads", type=int, default=16, help="CPU threads for training and evaluation")
a = p.parse_args()

for name in a.tasks:
    task, texts, labels, test_texts, test_gold = common.load(name)
    print(f"{name}: {len(texts)} gold texts, {len(test_texts)} test texts")
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in common.SEEDS:
            torch.set_num_threads(a.threads)
            out = Path(tmp) / str(seed)
            t0 = time.perf_counter()
            train_student(
                texts, None, task.labels, out, a.student, labels, 0.0, a.epochs, seed=seed, log=lambda _: None
            )
            seconds = time.perf_counter() - t0
            model, tok = load_student(out, "cpu")
            pred = student_logits(model, tok, test_texts).argmax(-1).tolist()
            runs.append(common.scores(pred, test_gold) | {"train_seconds": seconds})
            print(f"  seed {seed}: accuracy {runs[-1]['accuracy']:.3f} ({seconds:.0f}s)")
        onnx = Path(tmp) / "onnx"
        export_onnx(Path(tmp) / "0", onnx, int8=True)
        pr = Predictor(onnx, int8=True, threads=common.LATENCY_THREADS)
        int8 = common.scores(pr.logits(test_texts).argmax(-1).tolist(), test_gold)
        latency = common.latency_ms(lambda b, pr=pr: pr.logits(b, batch_size=1), test_texts)
        size_mb = dir_size_mb([onnx / "model.int8.onnx"])
    common.write(
        f"gold-only-{name}-{a.student.split('/')[-1]}",
        {
            "task": name,
            "method": "gold only",
            "model": a.student,
            "n_gold": len(texts),
            "n_test": len(test_texts),
            "seeds": list(common.SEEDS),
            "epochs": a.epochs,
            **common.over_seeds(runs),
            "int8_accuracy": int8["accuracy"],
            "latency_ms_b1": latency,
            "size_mb": size_mb,
            "deployed_as": "ONNX int8",
            "device": "cpu",
            "train_threads": a.threads,
            "latency_threads": common.LATENCY_THREADS,
        },
    )
