"""Choose the teacher setup on a dev split: every combination of teacher, few-shot examples and prompt is
scored on held-out gold-labelled training texts (never the test set), best first."""

from __future__ import annotations

import gc
import random
import time
from collections.abc import Callable

import torch

from .evaluate import accuracy, macro_f1, multilabel_metrics
from .student import few_shot_examples
from .task import Task
from .teacher import LabelScorer, load_teacher


def dev_split(texts: list[str], gold: list, size: int, seed: int) -> tuple[list[int], list[int]]:
    """(dev indices, the rest), among texts that have a gold label."""
    labelled = [i for i, g in enumerate(gold) if g not in (None, [])]
    rng = random.Random(seed + 1)  # not the stream that picks few-shot examples
    dev = sorted(rng.sample(labelled, min(size, len(labelled))))
    chosen = set(dev)
    return dev, [i for i in range(len(texts)) if i not in chosen]


def tune(
    tasks: list[tuple[str, Task]],
    texts: list[str],
    gold: list,
    teachers: list[str],
    shots: list[int],
    dev_size: int,
    device: str,
    load_4bit: bool,
    seed: int,
    log: Callable[[str], None],
) -> list[dict]:
    """One row per (teacher, prompt, shots): dev accuracy, macro-F1 and seconds per text."""
    dev, rest = dev_split(texts, gold, dev_size, seed)
    if not dev:
        raise SystemExit("tune needs gold labels in the training data")
    dev_texts, dev_gold = [texts[i] for i in dev], [gold[i] for i in dev]
    rows = []
    for name in teachers:  # one teacher in memory at a time
        log(f"loading teacher {name}")
        base = load_teacher(name, tasks[0][1], device, load_4bit)
        for prompt, task in tasks:
            for k in shots:
                if k and (task.multi_label or "{examples}" not in task.prompt):
                    log(f"skipping {k}-shot with {prompt}: the prompt has no {{examples}} line")
                    continue
                examples = (
                    few_shot_examples([texts[i] for i in rest], [gold[i] for i in rest], k, seed)
                    if k
                    else None
                )
                scorer = LabelScorer(
                    base.model, base.tok, task, max_tokens=base.max_tokens, examples=examples
                )
                t0 = time.perf_counter()
                scores = scorer.score_many(dev_texts)
                seconds = (time.perf_counter() - t0) / len(dev_texts)
                if task.multi_label:
                    pred = [frozenset(i for i, x in enumerate(s) if x > 0) for s in scores]
                    m = multilabel_metrics(pred, [frozenset(g) for g in dev_gold], len(task.labels))
                else:
                    pred = [max(range(len(s)), key=s.__getitem__) for s in scores]
                    m = {"accuracy": accuracy(pred, dev_gold), "macro_f1": macro_f1(pred, dev_gold)}
                row = {"teacher": name, "prompt": prompt, "few_shot": k, **m, "seconds_per_text": seconds}
                log(
                    f"{name} | {prompt} | {k}-shot: "
                    f"accuracy {row['accuracy']:.3f}, macro-F1 {row['macro_f1']:.3f}"
                )
                rows.append(row)
        del base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return sorted(rows, key=lambda r: (-r["accuracy"], -r["macro_f1"]))


def table(rows: list[dict], n_dev: int) -> str:
    head = (
        "| Teacher | Prompt | Few-shot examples per class | Dev accuracy | Dev macro-F1 "
        "| Seconds per text |\n"
        "|---|---|---|---|---|---|\n"
    )
    body = "\n".join(
        f"| `{r['teacher']}` | {r['prompt']} | {r['few_shot']} "
        f"| {r['accuracy']:.3f} | {r['macro_f1']:.3f} "
        f"| {r['seconds_per_text']:.3f} |"
        for r in rows
    )
    return (
        head
        + body
        + (
            f"\n\nScored on {n_dev} gold-labelled training texts held out from the few-shot examples "
            "(not the test set), best first.\n"
        )
    )
