"""Performance-guided distillation (PGKD, Di Palo et al. 2024): in rounds, find the label pairs the student
confuses most on held-out training texts, have the teacher write hard examples for them, and train again.

The held-out texts are a seeded share of the run's training texts, compared with the teacher's labels, which
are what the student learns. They are never trained on. The test set is only reported, never used to choose.
"""

from __future__ import annotations

import random

from .analyse import confusions
from .task import Task

PAIRS = 5  # the most frequent confusions targeted per round
SHOWN = 3  # misread held-out texts shown in a request


def hold_out(rows: list[dict], share: float, seed: int = 0) -> set[int]:
    """Row indices of a seeded share of the training texts; the windows of one document stay together."""
    key = [("doc", r["doc"]) if "doc" in r else ("row", i) for i, r in enumerate(rows)]
    docs = sorted(set(key))
    picked = set(random.Random(seed).sample(docs, max(1, round(share * len(docs)))))
    return {i for i, k in enumerate(key) if k in picked}


def jobs(task: Task, texts: list[str], pred: list[int], target: list[int]) -> list[tuple[int, str]]:
    """Requests for synth.synthesise: for each of the PAIRS most frequent confusions (a read as b), texts of a
    that could be taken for b and texts of b that could be taken for a (hard negatives), each shown up to
    SHOWN held-out texts that were misread that way."""
    found = {(g, p): idx for g, p, _, idx in confusions(pred, target, len(target))}  # most frequent first
    pairs = dict.fromkeys(ab for g, p in list(found)[:PAIRS] for ab in ((g, p), (p, g)))
    v = task.verbalisations
    out = []
    for a, b in pairs:
        extra = (
            f' Make every text a hard case: easy to mistake for the category "{v[b]}", yet clearly "{v[a]}" '
            "to a careful reader."
        )
        shown = [" ".join(texts[i].split()) for i in found.get((a, b), [])[:SHOWN]]
        if shown:
            extra += f' A classifier took texts like these, which are "{v[a]}", for "{v[b]}":\n'
            extra += "\n".join(f"- {t}" for t in shown)
        out.append((a, extra))
    return out


def table(rounds: list[dict]) -> str:
    """The rounds as a markdown table, and the round with the best held-out score."""
    fmt = lambda x: "" if x is None else f"{x:.3f}"  # noqa: E731
    lines = [
        "| round | texts added | training texts | held-out agreement | test accuracy |",
        "|---|---|---|---|---|",
        *(
            f"| {r['round']} | {r['added']} | {r['train']} | {fmt(r['held_out'])} | {fmt(r['test'])} |"
            for r in rounds
        ),
    ]
    best = max(rounds, key=lambda r: r["held_out"])
    return "\n".join(lines) + (
        f"\n\nHeld-out agreement: the student's agreement with the teacher on {rounds[0]['n_held_out']} "
        f"held-out training texts. Best by it: round {best['round']} ({best['dir']}). Round 0 is the run "
        "retrained without the held-out texts.\n"
    )
