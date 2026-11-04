"""Error analysis for a run: where the teacher and the student go wrong, as an HTML page and a short summary."""

from __future__ import annotations

import html
import json
import math
from collections import Counter
from pathlib import Path


def per_label(pred: list[int], gold: list[int], n: int) -> list[dict]:
    """Support, precision, recall and F1 of every label (single-label predictions)."""
    out = []
    for c in range(n):
        tp = sum(p == c and g == c for p, g in zip(pred, gold, strict=True))
        fp = sum(p == c and g != c for p, g in zip(pred, gold, strict=True))
        fn = sum(p != c and g == c for p, g in zip(pred, gold, strict=True))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out.append(
            {
                "support": tp + fn,
                "precision": prec,
                "recall": rec,
                "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            }
        )
    return out


def confusions(pred: list[int], gold: list[int], top: int) -> list[tuple[int, int, int, list[int]]]:
    """The most frequent (gold, predicted) mistakes: (gold, pred, count, indices of the texts)."""
    where: dict[tuple[int, int], list[int]] = {}
    for i, (p, g) in enumerate(zip(pred, gold, strict=True)):
        if p != g:
            where.setdefault((g, p), []).append(i)
    ranked = sorted(where.items(), key=lambda kv: -len(kv[1]))[:top]
    return [(g, p, len(idx), idx) for (g, p), idx in ranked]


def review(rows: list[dict], labels: list[str]) -> tuple[list[dict], int]:
    """A queue of training texts (rows of a single-label run's train.jsonl) for a person to check, and how many
    of them contradict their gold label. First the texts whose gold label differs from the teacher's top
    label, the teacher's most confident first: there the gold label is most likely wrong (or the prompt
    misleads the teacher). Then, for an ensemble (rows with ``teacher_scores``), the other texts on which its
    members' top labels differ, the most divided first (the smallest share of members on the most common
    label), then the least confident. Nothing here changes training."""

    def top(scores: list[float]) -> tuple[int, float]:
        m = max(scores)
        p = [math.exp(x - m) for x in scores]
        best = max(range(len(p)), key=p.__getitem__)
        return best, p[best] / sum(p)

    wrong, divided = [], []
    for r in rows:
        label, conf = top(r["scores"])
        members = [top(s)[0] for s in r.get("teacher_scores") or []]
        row = {
            "text": r["text"],
            "gold": "" if r["gold"] is None else labels[r["gold"]],
            "teacher_label": labels[label],
            "teacher_confidence": round(conf, 3),
            "member_labels": "; ".join(labels[m] for m in members),
        }
        agree = max(Counter(members).values()) / len(members) if members else 1.0
        if r["gold"] is not None and r["gold"] != label:
            wrong.append((-conf, row))
        elif agree < 1:
            divided.append(((agree, conf), row))
    wrong.sort(key=lambda x: x[0])
    divided.sort(key=lambda x: x[0])
    return [row for _, row in wrong + divided], len(wrong)


CSS = """
  :root { --bg: #fbfbfa; --fg: #1d1d1b; --muted: #5f5e5a; --line: #dcdbd6; --bad: #b3261e; }
  @media (prefers-color-scheme: dark) { :root { --bg: #161615; --fg: #efeee9; --muted: #b3b2aa; --line: #393935; --bad: #f2b8b5; } }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; }
  main { max-width: 1000px; margin: 0 auto; padding: 32px 16px; }
  h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 32px; }
  p.sub, td.m { color: var(--muted); }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
  td.low { color: var(--bad); font-weight: 600; }
  .ex { color: var(--muted); font-size: .9rem; }
  .scroll { overflow-x: auto; }
"""


def predictions(run: Path) -> dict:
    """For the gold-labelled test texts of a single-label run: the task, texts, gold labels, and the teacher's
    and the deployed student's predicted labels."""
    from .export import Predictor
    from .task import load_task

    task = load_task(run / "task.yaml")
    if task.multi_label:
        raise SystemExit("this view is for single-label tasks; see results.md for multi-label runs")
    rows = [
        r
        for r in map(json.loads, (run / "test.jsonl").read_text(encoding="utf-8").splitlines())
        if r.get("gold") is not None
    ]
    if not rows:
        raise SystemExit(f"{run}/test.jsonl has no gold labels")
    texts = [r["text"] for r in rows]
    n = len(task.labels)
    return {
        "task": task,
        "texts": texts,
        "gold": [r["gold"] for r in rows],
        "teacher": [max(range(n), key=r["scores"].__getitem__) for r in rows],
        "student": [int(i) for i in Predictor(run / "onnx").logits(texts).argmax(-1)],
    }


def build(run: Path, top: int = 15) -> tuple[str, str]:
    """(HTML page, short Markdown summary) for a single-label run with a gold-labelled test set."""
    p = predictions(run)
    task, texts, gold, teacher, student = p["task"], p["texts"], p["gold"], p["teacher"], p["student"]
    names = task.labels

    t_lab, s_lab = per_label(teacher, gold, len(names)), per_label(student, gold, len(names))
    order = sorted(range(len(names)), key=lambda c: t_lab[c]["f1"])  # the teacher's weakest labels first
    disagree = [i for i in range(len(texts)) if teacher[i] != student[i]]
    t_acc = sum(t == g for t, g in zip(teacher, gold, strict=True)) / len(gold)
    s_acc = sum(s == g for s, g in zip(student, gold, strict=True)) / len(gold)
    e = html.escape

    def f(x: float) -> str:
        return f"{x:.2f}"

    def examples(idx: list[int], k: int = 2) -> str:
        return "<br>".join(f"&ldquo;{e(texts[i][:140])}&rdquo;" for i in idx[:k])

    label_rows = "\n".join(
        f"<tr><td>{e(names[c])}</td><td class='m'>{t_lab[c]['support']}</td>"
        + "".join(
            f"<td class='{'low' if lab[c]['f1'] < 0.5 else ''}'>{f(lab[c][k])}</td>"
            for lab in (t_lab, s_lab)
            for k in ("precision", "recall", "f1")
        )
        + "</tr>"
        for c in order
    )

    def confusion_rows(pred: list[int]) -> str:
        return "\n".join(
            f"<tr><td>{e(names[g])}</td><td>{e(names[p])}</td><td>{n}</td><td class='ex'>{examples(idx)}</td></tr>"
            for g, p, n, idx in confusions(pred, gold, top)
        )

    dis = "\n".join(
        f"<tr><td class='ex'>&ldquo;{e(texts[i][:160])}&rdquo;</td><td>{e(names[gold[i]])}</td>"
        f"<td>{e(names[teacher[i]])}</td><td>{e(names[student[i]])}</td></tr>"
        for i in disagree[:top]
    )
    head = "<tr><th>Gold</th><th>Predicted</th><th>Texts</th><th>Examples</th></tr>"
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Error analysis: {e(task.name)}</title><style>{CSS}</style></head>
<body><main>
<h1>Error analysis: {e(task.name)}</h1>
<p class="sub">{len(texts)} gold-labelled test texts. Teacher accuracy {t_acc:.3f}, student (deployed model)
accuracy {s_acc:.3f}; they disagree on {len(disagree)} texts. Written by <code>tinydistill analyse</code>.</p>

<h2>Per label, the teacher's weakest first</h2>
<p class="sub">F1 below 0.5 is highlighted. A label the teacher gets wrong often is the first place to look: a
clearer name, a description, or merging it with the label it is confused with usually helps more than anything
on the student side.</p>
<div class="scroll"><table>
<tr><th>Label</th><th>Texts</th><th>Teacher precision</th><th>Teacher recall</th><th>Teacher F1</th>
<th>Student precision</th><th>Student recall</th><th>Student F1</th></tr>
{label_rows}
</table></div>

<h2>The teacher's most common mistakes</h2>
<div class="scroll"><table>{head}{confusion_rows(teacher)}</table></div>

<h2>The student's most common mistakes</h2>
<div class="scroll"><table>{head}{confusion_rows(student)}</table></div>

<h2>Texts where the student and the teacher disagree</h2>
<div class="scroll"><table><tr><th>Text</th><th>Gold</th><th>Teacher</th><th>Student</th></tr>{dis}</table></div>
</main></body></html>
"""
    worst = [names[c] for c in order[:5]]
    pairs = Counter({(names[g], names[p]): n for g, p, n, _ in confusions(teacher, gold, 5)})
    summary = (
        f"Teacher accuracy {t_acc:.3f}, student {s_acc:.3f}; they disagree on {len(disagree)} of {len(texts)} texts.\n"
        f"The teacher's weakest labels: {', '.join(worst)}.\n"
        "Its most common mistakes: "
        + "; ".join(f"{g} -> {p} ({n})" for (g, p), n in pairs.most_common(5))
        + "."
    )
    return page, summary
