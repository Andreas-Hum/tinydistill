"""A second student round on a finished run: for each seed, a new student is trained as the run's was (same
student model, epochs, learning rate, batch size, held-out share, gold labels per class and seed) but on other
targets, built from the teacher's soft labels and the same seed's round-1 student (the run's model):

- teacher: the teacher's soft labels again, as in round 1 (the control: round 2 differs only in its targets);
- mix_0.5: per training text, the mean of the teacher's distribution and the round-1 student's;
- student_only: the round-1 student's distributions alone (born-again networks, Furlanello et al., arXiv
  1805.04770);
- relabel: the teacher's soft labels, except on the training texts where the round-1 student's top label is
  another one and its probability is at least 0.9: there the student's distribution (student-side label
  denoising, as in PA-KD, arXiv 2312.10185, and CanDist, arXiv 2506.03857).

The cross-entropy on the K gold labels per class stays as in round 1. The round-2 students are written to a
temporary directory and evaluated on CPU on the run's gold-labelled test texts, as `tinydistill eval` does.
The output (JSON) gives, per variant, test accuracy, macro-F1 and agreement with the teacher (mean, standard
deviation and per seed), and the number of training texts whose target's top label differs from the
teacher's; and the round-1 student's numbers from the run's results.json. The run is only read:

    python examples/students/round2.py runs/banking77-7b-similar16-probe --device cuda \\
        > examples/students/round2-banking77.json

The first seed is the run's own seed (taken from its seeds/ directory; 0 if the run has a single seed).
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

from tinydistill.cli import _seed_dirs
from tinydistill.evaluate import accuracy, aggregate, macro_f1
from tinydistill.student import few_gold, load_student, student_logits, train_student
from tinydistill.task import load_task
from tinydistill.teacher import pick_device

VARIANTS = ("teacher", "mix_0.5", "student_only", "relabel")
RELABEL_CONFIDENCE = 0.9


def targets(variant: str, teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    """Round-2 targets (texts, labels) from the teacher's and the round-1 student's distributions."""
    if variant == "teacher":
        return teacher
    if variant == "mix_0.5":
        return (teacher + student) / 2
    if variant == "student_only":
        return student
    p, top = student.max(-1)
    swap = (top != teacher.argmax(-1)) & (p >= RELABEL_CONFIDENCE)
    return torch.where(swap[:, None], student, teacher)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def round2(run: Path, device: str) -> dict:
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    task = load_task(run / "task.yaml")
    unsupported = [
        k for k in ("warmup_pairs", "init_head", "trim_vocab", "chunk_tokens", "gold_file") if meta.get(k)
    ]
    if task.multi_label or unsupported:
        raise SystemExit(f"{run}: not supported here: {'multi-label' if task.multi_label else unsupported}")
    rows = read_jsonl(run / "train.jsonl")
    texts = [r["text"] for r in rows]
    teacher = torch.tensor([r["scores"] for r in rows], dtype=torch.float64).softmax(-1).float()
    dirs = _seed_dirs(run, "student")
    first = int(dirs[1].parent.name) - 1 if len(dirs) > 1 else 0  # further seeds live in seeds/<seed>/
    k = meta.get("gold_per_class")
    gold = few_gold([r["gold"] for r in rows], k, first) if k else None  # as cmd_train: same for all seeds
    test = [r for r in read_jsonl(run / "test.jsonl") if r["gold"] is not None]
    test_texts, test_gold = [r["text"] for r in test], [r["gold"] for r in test]
    teacher_test = [max(range(len(r["scores"])), key=r["scores"].__getitem__) for r in test]

    per_seed: dict[str, list[dict]] = {v: [] for v in VARIANTS}
    for i, d in enumerate(dirs):
        seed = first + i
        model, tok = load_student(d, device)
        max_length = getattr(model.config, "tinydistill_max_length", None) or 128
        student = student_logits(model, tok, texts).softmax(-1)
        del model
        for v in VARIANTS:
            t = targets(v, teacher, student)
            log(f"{run.name}, seed {seed}: round 2, {v}")
            with tempfile.TemporaryDirectory() as tmp:
                kept = train_student(
                    texts,
                    t,
                    task.labels,
                    Path(tmp),
                    meta["student"],
                    gold,
                    meta["ce_weight"],
                    meta["epochs"],
                    meta["batch_size"],
                    meta["lr"],
                    max_length,
                    device,
                    seed,
                    log,
                    val_share=meta.get("val_share") or 0.0,
                    patience=meta.get("patience") or 3,
                    outside=task.outside_id,
                )
                m2, tok2 = load_student(Path(tmp), "cpu")
                pred = student_logits(m2, tok2, test_texts).argmax(-1).tolist()
            per_seed[v].append(
                {
                    "accuracy": accuracy(pred, test_gold),
                    "macro_f1": macro_f1(pred, test_gold),
                    "agreement": accuracy(pred, teacher_test),
                    "targets_changed": int((t.argmax(-1) != teacher.argmax(-1)).sum()),
                    "kept_epoch": kept,
                }
            )
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    row = next(m for m in results["models"] if m["name"] == "student, PyTorch")
    keys = [f"{m}{s}" for m in ("accuracy", "macro_f1", "agreement") for s in ("", "_std", "_seeds")]
    return {
        "run": run.as_posix(),
        "student": meta["student"],
        "epochs": meta["epochs"],
        "seeds": [first + i for i in range(len(dirs))],
        "n_train": len(texts),
        "n_gold": sum(g is not None for g in gold) if gold else 0,
        "n_test": len(test),
        "relabel_confidence": RELABEL_CONFIDENCE,
        "round_1": {key: row[key] for key in keys if key in row},
        "round_2": {v: aggregate(per_seed[v]) for v in VARIANTS},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", help="a finished run directory (labelled, trained and evaluated)")
    parser.add_argument("--device", default="auto", help="device for training: cuda, cpu or auto")
    a = parser.parse_args()
    print(json.dumps(round2(Path(a.run), pick_device(a.device)), indent=1))
