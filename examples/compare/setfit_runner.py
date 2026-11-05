"""SetFit (Tunstall et al., 2022) on a task's 8 gold labels per class, the same texts tinydistill's runs use,
with three training seeds, on the CPU. Writes examples/compare/results/setfit-<task>-<body>.json.

Needs setfit in its own environment (see examples/compare/run.sh). Training: one epoch of contrastive pairs
with 20 pair iterations, then a logistic-regression head, as in examples/banking77/setfit_baseline.py.

    python examples/compare/setfit_runner.py banking77 --body MiniLM-L6 --threads 16
"""

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import common
import numpy as np
import torch
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

BODIES = {
    "MiniLM-L6": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet-base": "sentence-transformers/all-mpnet-base-v2",
    "multilingual-MiniLM-L12": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}

p = argparse.ArgumentParser()
p.add_argument("tasks", nargs="+", choices=list(common.TASKS))
p.add_argument(
    "--body", action="append", choices=list(BODIES), help="repeatable; default MiniLM-L6, mpnet-base"
)
p.add_argument("--threads", type=int, default=16, help="CPU threads for training and evaluation")
a = p.parse_args()

for name in a.tasks:
    task, texts, labels, test_texts, test_gold = common.load(name)
    print(f"{name}: {len(texts)} gold texts, {len(test_texts)} test texts")
    for short in a.body or ["MiniLM-L6", "mpnet-base"]:
        runs, models = [], []
        for seed in common.SEEDS:
            torch.set_num_threads(a.threads)
            model = SetFitModel.from_pretrained(BODIES[short], device="cpu")
            out = tempfile.mkdtemp()
            args = TrainingArguments(
                batch_size=32, num_epochs=1, num_iterations=20, seed=seed, output_dir=out
            )
            trainer = Trainer(
                model=model, args=args, train_dataset=Dataset.from_dict({"text": texts, "label": labels})
            )
            t0 = time.perf_counter()
            trainer.train()
            seconds = time.perf_counter() - t0
            shutil.rmtree(out, ignore_errors=True)
            pred = [int(x) for x in np.asarray(model.predict(test_texts, batch_size=64, as_numpy=True))]
            runs.append(common.scores(pred, test_gold) | {"train_seconds": seconds})
            models.append(model)
            print(f"  SetFit {short}, seed {seed}: accuracy {runs[-1]['accuracy']:.3f} ({seconds:.0f}s)")
        model = models[0]  # timed and sized: the first seed's model, with the library's own predict
        torch.set_num_threads(common.LATENCY_THREADS)
        latency = common.latency_ms(lambda b, m=model: m.predict(b, as_numpy=True), test_texts)
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)
            size_mb = sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file()) / 1e6
        common.write(
            f"setfit-{name}-{short}",
            {
                "task": name,
                "method": "SetFit",
                "model": BODIES[short],
                "short": short,
                "n_gold": len(texts),
                "n_test": len(test_texts),
                "seeds": list(common.SEEDS),
                **common.over_seeds(runs),
                "latency_ms_b1": latency,
                "size_mb": size_mb,
                "device": "cpu",
                "train_threads": a.threads,
                "latency_threads": common.LATENCY_THREADS,
                "settings": "1 epoch, 20 pair iterations, batch 32, logistic-regression head",
            },
        )
        del models, model
