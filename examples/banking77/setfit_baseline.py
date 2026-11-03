"""SetFit on Banking77 with the same 8 gold labels per class that the tinydistill runs use, as a comparison.

SetFit (Tunstall et al., 2022) is a strong few-shot method that needs no LLM: it fine-tunes a sentence encoder
contrastively on pairs of the labelled texts, then fits a logistic-regression head. Here it gets the same
616 labelled texts (8 per class, picked by tinydistill's own function with the same seed) and two bodies:
MiniLM-L6, the encoder of the tinydistill student, and mpnet-base, the encoder SetFit's paper recommends. Each
runs with three seeds. Needs `pip install setfit` (its own environment; see examples/banking77/run.sh).

Writes examples/banking77/setfit.json and examples/banking77/setfit.md (the comparison table).
"""

import json
import statistics
import time
from pathlib import Path

import numpy as np
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

from tinydistill.data import read_examples
from tinydistill.evaluate import accuracy, macro_f1
from tinydistill.student import few_gold
from tinydistill.task import load_task

HERE = Path(__file__).parent
BODIES = {
    "MiniLM-L6": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet-base": "sentence-transformers/paraphrase-mpnet-base-v2",
}
SEEDS = (0, 1, 2)

task = load_task(HERE / "task.yaml")
train = list(read_examples("hf:PolyAI/banking77:train", task))
test = list(read_examples("hf:PolyAI/banking77:test", task))
gold = few_gold([e.gold for e in train], 8, 0)  # the same 616 texts as --gold-per-class 8 --seed 0
picked = [(e.text, g) for e, g in zip(train, gold, strict=True) if g is not None]
texts, labels = [t for t, _ in picked], [g for _, g in picked]
test_texts, test_gold = [e.text for e in test], [e.gold for e in test]
print(f"{len(picked)} labelled training texts, {len(test)} test texts")

results = {"n_labelled": len(picked), "n_test": len(test), "seeds": list(SEEDS), "bodies": {}}
for short, body in BODIES.items():
    runs = []
    for seed in SEEDS:
        model = SetFitModel.from_pretrained(body)
        args = TrainingArguments(
            batch_size=32, num_epochs=1, num_iterations=20, seed=seed, output_dir="runs/setfit-checkpoints"
        )
        trainer = Trainer(
            model=model, args=args, train_dataset=Dataset.from_dict({"text": texts, "label": labels})
        )
        t0 = time.perf_counter()
        trainer.train()
        seconds = time.perf_counter() - t0
        pred = [int(p) for p in np.asarray(model.predict(test_texts, batch_size=64, as_numpy=True))]
        runs.append(
            {
                "accuracy": accuracy(pred, test_gold),
                "macro_f1": macro_f1(pred, test_gold),
                "train_seconds": seconds,
            }
        )
        print(f"SetFit {short}, seed {seed}: accuracy {runs[-1]['accuracy']:.3f} ({seconds:.0f}s)")
    results["bodies"][short] = {"model": body, "runs": runs}
(HERE / "setfit.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def pm(values: list[float]) -> str:
    return f"{statistics.fmean(values):.3f} ± {statistics.stdev(values):.3f}"


def row(path: Path, name: str) -> dict:
    return next(m for m in json.loads(path.read_text(encoding="utf-8"))["models"] if m["name"] == name)


teacher_run = json.loads((HERE / "banking77-7b-8shot.json").read_text(encoding="utf-8"))
teacher = next(m for m in teacher_run["models"] if m["role"] == "teacher")
upper = next(m for m in teacher_run["models"] if m["role"] == "upper bound")
student = row(HERE / "students" / "all-MiniLM-L6-v2-20-epochs.json", "student, PyTorch")
static = row(HERE / "students" / "static.json", "student, PyTorch")
seeded = lambda m, k: f"{m[k]:.3f} ± {m[k + '_std']:.3f}" if f"{k}_std" in m else f"{m[k]:.3f}"  # noqa: E731
lines = [
    "| Method | Gold labels | Also uses | Accuracy | Macro-F1 |",
    "|---|---|---|---|---|",
    f"| Qwen2.5-7B-Instruct, 8 examples per class in the prompt | 616 | | {teacher['accuracy']:.3f} "
    f"| {teacher['macro_f1']:.3f} |",
]
for short, r in results["bodies"].items():
    acc = [x["accuracy"] for x in r["runs"]]
    f1 = [x["macro_f1"] for x in r["runs"]]
    lines.append(f"| SetFit, {short} | 616 | | {pm(acc)} | {pm(f1)} |")
lines += [
    f"| tinydistill, MiniLM-L6 student | 616 | the 7B teacher's soft labels on 10,003 texts "
    f"| {seeded(student, 'accuracy')} | {seeded(student, 'macro_f1')} |",
    f"| tinydistill, static student | 616 | the same | {seeded(static, 'accuracy')} "
    f"| {seeded(static, 'macro_f1')} |",
    f"| MiniLM-L6 trained on all gold labels (upper bound) | 10,003 | | {seeded(upper, 'accuracy')} "
    f"| {seeded(upper, 'macro_f1')} |",
]
note = (
    f"\nBanking77, {results['n_test']} test texts; mean ± standard deviation over three seeds. Every method "
    "sees the same 616 gold-labelled texts (8 per class). SetFit: one epoch of contrastive training with 20 "
    "pair iterations, then a logistic-regression head (the library's defaults otherwise). The tinydistill "
    "students train for 20 epochs; the upper bound for 5.\n"
)
(HERE / "setfit.md").write_text("\n".join(lines) + "\n" + note, encoding="utf-8")
print((HERE / "setfit.md").read_text(encoding="utf-8"))
