"""CLINC150 scored as the few-shot papers score it: the 4,500 in-scope test texts, 150 intents.

Our CLINC150 runs have 151 labels, the 150 intents plus "oos" (outside all of them), and 5,500 test texts. The
papers (Zhang et al. 2021, arXiv 2109.06349; Milios et al. 2023, arXiv 2309.10954) train and test on the 150
intents alone. This script takes saved runs and scores their teacher and students on the in-scope test texts:

- accuracy_150: the best of the 150 intents, as a model trained without "oos" would answer;
- accuracy_151: the model's own answer, so an in-scope text it calls "oos" counts as wrong.

    python examples/clinc150/in_scope.py runs/clinc150-gemma4-12b-similar16-probe \
        runs/clinc150-gemma4-12b-probe-only-similar16 runs/clinc150-students/all-MiniLM-L12-v2 \
        runs/clinc150-students/all-mpnet-base-v2 > examples/clinc150/in-scope.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tinydistill.export import Predictor


def student_logits(path: Path, texts: list[str]) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(texts), 64):
            enc = tok(texts[b : b + 64], padding=True, truncation=True, max_length=128, return_tensors="pt")
            out.append(model(**enc).logits.numpy())
    return np.concatenate(out)


def scores(z: np.ndarray, gold: np.ndarray, oos: int) -> dict:
    inside = z.copy()
    inside[:, oos] = -np.inf
    return {
        "accuracy_150": float((inside.argmax(-1) == gold).mean()),
        "accuracy_151": float((z.argmax(-1) == gold).mean()),
    }


def mean_std(per_seed: list[dict]) -> dict:
    out: dict = {}
    for k in per_seed[0]:
        xs = [s[k] for s in per_seed]
        out[k], out[k + "_std"], out[k + "_seeds"] = float(np.mean(xs)), float(np.std(xs)), xs
    return out


def score_run(run: Path) -> dict:
    labels = json.loads((run / "onnx" / "labels.json").read_text(encoding="utf-8"))
    oos = labels.index("oos")
    rows = [json.loads(line) for line in (run / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["gold"] != oos]
    texts, gold = [r["text"] for r in rows], np.array([r["gold"] for r in rows])
    seeds = [run / "student"] + sorted((run / "seeds").glob("*/student"))
    return {
        "run": run.name,
        "n_test": len(rows),
        "teacher": scores(np.array([r["scores"] for r in rows]), gold, oos),
        "student": mean_std([scores(student_logits(s, texts), gold, oos) for s in seeds]),
        "student_int8": scores(Predictor(run / "onnx", int8=True, threads=2).logits(texts), gold, oos),
    }


if __name__ == "__main__":
    print(json.dumps([score_run(Path(a)) for a in sys.argv[1:]], indent=1))
