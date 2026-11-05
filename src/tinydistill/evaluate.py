"""Quality metrics, latency benchmarks and the results table."""

from __future__ import annotations

import os
import platform
import statistics
import time
from collections.abc import Callable, Hashable, Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path

import torch


def accuracy(pred: Sequence[object], gold: Sequence[object]) -> float:
    return sum(p == g for p, g in zip(pred, gold, strict=True)) / len(gold) if gold else float("nan")


def macro_f1(pred: Sequence[Hashable], gold: Sequence[Hashable]) -> float:
    """Unweighted mean F1 over the classes that occur in gold or pred (as scikit-learn's 'macro')."""
    classes = set(gold) | set(pred)
    if not classes:
        return float("nan")
    f1s = []
    for c in classes:
        tp = sum(p == c and g == c for p, g in zip(pred, gold, strict=True))
        fp = sum(p == c and g != c for p, g in zip(pred, gold, strict=True))
        fn = sum(p != c and g == c for p, g in zip(pred, gold, strict=True))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / len(f1s)


def multilabel_metrics(
    pred: Sequence[AbstractSet[int]], gold: Sequence[AbstractSet[int]], labels: int
) -> dict:
    """Exact-match accuracy, macro-F1 (mean over labels) and micro-F1 for sets of labels."""
    tp = [0] * labels
    fp = [0] * labels
    fn = [0] * labels
    for p, g in zip(pred, gold, strict=True):
        for c in p & g:
            tp[c] += 1
        for c in p - g:
            fp[c] += 1
        for c in g - p:
            fn[c] += 1
    f1 = [2 * t / (2 * t + a + b) if t else 0.0 for t, a, b in zip(tp, fp, fn, strict=True) if t + a + b]
    micro = 2 * sum(tp) / (2 * sum(tp) + sum(fp) + sum(fn)) if sum(tp) else 0.0
    return {
        "accuracy": sum(p == g for p, g in zip(pred, gold, strict=True)) / len(gold)
        if gold
        else float("nan"),
        "macro_f1": sum(f1) / len(f1) if f1 else float("nan"),
        "micro_f1": micro,
    }


def ordinal_metrics(pred: Sequence[int], gold: Sequence[int]) -> dict:
    """For ordered labels: mean absolute error in label steps, and the share at most one step off."""
    d = [abs(p - g) for p, g in zip(pred, gold, strict=True)]
    return {"mae": sum(d) / len(d), "within_one": sum(x <= 1 for x in d) / len(d)} if d else {}


def outside_metrics(pred: Sequence[int], gold: Sequence[int], outside: int) -> dict:
    """For a task with an `outside` label (index ``outside``), when the gold labels contain it: the accuracy
    on the texts whose gold label is another one (a prediction of `outside` counts as wrong there), and the
    recall and precision of `outside` (0 without a correct `outside` prediction, as in :func:`macro_f1`)."""
    if outside not in gold:
        return {}
    inside = [(p, g) for p, g in zip(pred, gold, strict=True) if g != outside]
    tp = sum(p == g == outside for p, g in zip(pred, gold, strict=True))
    return {
        "in_scope_accuracy": accuracy([p for p, _ in inside], [g for _, g in inside]),
        "outside_recall": tp / sum(g == outside for g in gold) if tp else 0.0,
        "outside_precision": tp / sum(p == outside for p in pred) if tp else 0.0,
    }


def time_calls(fn: Callable[[list[str]], object], texts: list[str], batch: int, reps: int) -> float:
    """Median wall time (ms) of fn on `reps` different batches of `batch` texts, after one warm-up call."""

    def batch_at(i: int) -> list[str]:
        return [texts[(i * batch + j) % len(texts)] for j in range(batch)]

    fn(batch_at(reps))  # warm-up on a batch that is not timed
    times = []
    for i in range(reps):
        b = batch_at(i)
        t0 = time.perf_counter()
        fn(b)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def bench(fn: Callable[..., object], texts: list[str], reps1: int, reps32: int = 0) -> dict:
    """Latency at batch 1 and (if reps32 > 0) batch 32. Throughput comes from batch 32 when measured,
    otherwise from batch 1 (for models that process one text at a time)."""
    out = {"latency_ms_b1": time_calls(fn, texts, 1, reps1)}
    if reps32:
        out["latency_ms_b32"] = time_calls(fn, texts, 32, reps32)
        out["throughput_per_s"] = 32 / (out["latency_ms_b32"] / 1000)
    else:
        out["throughput_per_s"] = 1000 / out["latency_ms_b1"]
    return out


def top_k_accuracy(scores: list[list[float]], gold: list[int], k: int = 3) -> float:
    """Share of texts whose gold label is among the k highest-scoring labels."""
    hits = sum(g in sorted(range(len(s)), key=s.__getitem__)[-k:] for s, g in zip(scores, gold, strict=True))
    return hits / len(gold) if gold else float("nan")


def aggregate(per_seed: list[dict]) -> dict:
    """Mean of each metric over seeds; with several seeds also the sample standard deviation and values."""
    if not per_seed or not per_seed[0]:
        return {}
    out: dict = {}
    for k in per_seed[0]:
        vals = [q[k] for q in per_seed]
        out[k] = statistics.fmean(vals)
        if len(vals) > 1:
            out[f"{k}_std"] = statistics.stdev(vals)
            out[f"{k}_seeds"] = vals
    return out


def dir_size_mb(paths: Sequence[Path]) -> float:
    return sum(p.stat().st_size for p in paths if p.exists()) / 1e6


def hf_weights_mb(name: str) -> float | None:
    """Size of a Hub model's weight files in the local cache (or of a local directory)."""
    p = Path(name)
    if not p.exists():
        try:
            from huggingface_hub import snapshot_download

            p = Path(snapshot_download(name, local_files_only=True))
        except Exception:
            return None
    files = list(p.glob("*.safetensors")) or list(p.glob("*.bin"))
    return dir_size_mb(files) if files else None


def environment(threads: int) -> dict:
    return {
        "cpu": platform.processor() or platform.machine(),
        "cpu_threads": threads,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def _fmt(v, spec: str) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return format(v, spec)


def _pm(row: dict, key: str) -> str:
    """A metric, with its standard deviation over seeds when there is one."""
    std = row.get(f"{key}_std")
    return _fmt(row.get(key), ".3f") + (f" ± {std:.3f}" if std is not None else "")


EXTRA_COLUMNS = (
    ("top3_accuracy", "Top-3 accuracy"),
    ("micro_f1", "Micro-F1"),
    ("mae", "MAE (label steps)"),
    ("within_one", "Within one step"),
    ("in_scope_accuracy", "In-scope accuracy"),
    ("outside_recall", "Out-of-scope recall"),
    ("outside_precision", "Out-of-scope precision"),
)


def markdown_table(results: dict) -> str:
    rows = results["models"]
    ref = next((r for r in rows if r["role"] == "teacher" and r.get("latency_ms_b1")), None)
    # extra quality columns, when the task has them: multi-label (micro-F1) or ordered labels (distances)
    extra = [(k, name) for k, name in EXTRA_COLUMNS if any(k in r for r in rows)]
    head = (
        "| Model | Device | Accuracy | Macro-F1 "
        + "".join(f"| {name} " for _, name in extra)
        + "| Agreement with teacher | Latency, batch 1 (ms) "
        "| Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |\n"
        + "|---" * (10 + len(extra))
        + "|\n"
    )
    lines = []
    for r in rows:
        speed = ref["latency_ms_b1"] / r["latency_ms_b1"] if ref and r.get("latency_ms_b1") else None
        lines.append(
            f"| {r['name']} | {r['device']} "
            f"| {_pm(r, 'accuracy')} | {_pm(r, 'macro_f1')} "
            + "".join(f"| {_pm(r, k)} " for k, _ in extra)
            + f"| {_pm(r, 'agreement')} | {_fmt(r.get('latency_ms_b1'), '.1f')} "
            f"| {_fmt(r.get('latency_ms_b32'), '.1f')} | {_fmt(r.get('throughput_per_s'), '.1f')} "
            f"| {_fmt(r.get('size_mb'), '.0f')} | {_fmt(speed, '.1f') + 'x' if speed else 'n/a'} |"
        )
    ds = results["data"]
    tr = results.get("training") or {}
    k = tr.get("gold_per_class") or 0
    sources = [f"{k} per class"] if k else []
    if tr.get("n_gold_file"):
        sources.append(f"{tr['n_gold_file']} from {tr.get('gold_file')}")
    trained = (
        f"The student learns from the teacher's soft labels on all {ds['n_train']} training texts, "
        f"plus cross-entropy on {tr.get('n_gold')} gold-labelled texts "
        f"({', '.join(sources)}; weight {tr.get('ce_weight')})."
        if sources
        else f"The student learns only from the teacher's soft labels on {ds['n_train']} "
        "unlabelled training texts."
    )
    if tr.get("few_shot_similar") and (tr.get("few_shot_sets") or 1) > 1:
        k, m = tr["few_shot_sets"], tr["few_shot_similar"]
        trained = (
            f"The teacher scores each text {k} times, each prompt showing it another {m} of its {k * m} "
            f"most similar of {tr.get('n_examples')} gold training examples, and averages the label "
            "probabilities. " + trained
        )
    elif tr.get("few_shot_similar"):
        trained = (
            f"The teacher's prompt shows each text the {tr['few_shot_similar']} most similar of "
            f"{tr.get('n_examples')} gold training examples. " + trained
        )
    elif tr.get("few_shot"):
        trained = (
            f"The teacher's prompt shows {tr['few_shot']} gold training examples per class "
            f"({tr.get('n_examples')} texts). " + trained
        )
    if probe := tr.get("probe"):
        dev = probe.get("dev")
        trained = (
            (
                "The teacher is"
                if probe.get("only")
                else "The teacher's probabilities are averaged with those of"
            )
            + f" a linear probe on its hidden states, fit on {probe['n_gold']} gold training texts"
            + (
                f"; fit on {dev['n_fit']} of them, teacher accuracy on the other {dev['n_dev']} is "
                f"{dev['in_context']:.1%} in context, {dev['probe']:.1%} for the probe and "
                f"{dev['combined']:.1%} combined. "
                if dev
                else ". "
            )
            + trained
        )
    if tr.get("gold_only"):
        trained += (
            f" The baseline is the same student trained with cross-entropy on those {tr.get('n_gold')} gold "
            "labels only, without the teacher."
        )
    if tr.get("upper_bound"):
        trained += (
            " The upper bound is the same student trained with cross-entropy on all "
            f"{tr.get('n_gold_upper_bound')} gold training labels and no teacher; "
            "its speed equals the student's."
        )
    if tr.get("val_share"):
        kept = sorted({e for runs in (tr.get("kept_epochs") or {}).values() for e in runs})
        trained += (
            f" Each student holds out {tr['val_share']:.0%} of its training texts and keeps the epoch "
            f"with the lowest loss on them (up to {tr.get('epochs')} epochs, stopping after "
            f"{tr.get('patience')} without improvement)"
            + (f"; kept epochs: {', '.join(map(str, kept))}." if kept else ".")
        )
    if tr.get("shortlist"):
        recall = tr.get("shortlist_recall")
        keep = tr.get("shortlist_keep")
        trained += (
            f" The teacher scores only the {tr['shortlist']} labels most similar to each text"
            + (f" and always {', '.join(keep)}" if keep else "")
            + (
                f"; the gold label was among them for {recall:.1%} of test texts."
                if recall is not None
                else "."
            )
        )
    if tr.get("chunk_tokens"):
        trained += (
            f" Texts longer than {tr['chunk_tokens']} tokens are read in windows of that many tokens, "
            f"{tr.get('chunk_stride')} apart: the training texts are windows, and a test text's prediction "
            "is the mean of its windows' probabilities."
        )
    if any("outside_recall" in r for r in rows):
        trained += (
            " In-scope accuracy is the accuracy on the test texts whose gold label is not the `outside` "
            "label (texts that fit none of the others); out-of-scope recall and precision are those of that "
            "label."
        )
    if (tr.get("seeds") or 1) > 1:
        trained += (
            f" Values with ± are the mean and standard deviation over {tr['seeds']} training seeds; "
            "the ONNX rows are the exported first-seed model."
        )
    notes = [
        "",
        f"Task `{results['task']}`, {ds['n_test']} gold-labelled test texts. "
        f"Teacher: `{results['teacher']}`. Student: `{results['student']}`. {trained}",
        f"CPU rows use {results['environment']['cpu_threads']} threads. "
        "Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' "
        "model files. "
        "The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes "
        "under three minutes, otherwise its throughput is 1 / batch-1 latency. "
        + (f"Speed-up is relative to the {ref['name']} on {ref['device']}." if ref else ""),
    ]
    return head + "\n".join(lines) + "\n" + "\n".join(notes) + "\n"


def compare_table(runs: list[tuple[str, dict]]) -> str:
    """One row per run: the student, its quality (mean over seeds) and the deployed model's speed and size
    (the int8 ONNX model when there is one, else fp32 ONNX, else PyTorch)."""
    head = (
        "| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 "
        "| Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) "
        "| Size (MB) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for name, res in runs:
        rows = [m for m in res["models"] if m["role"] == "student"]
        if not rows:
            continue
        quality = rows[0]  # the PyTorch row carries the per-seed quality
        teacher: dict = next((m for m in res["models"] if m["role"] == "teacher"), {})
        onnx = [m for m in rows if "ONNX" in m["name"]]
        deployed = next((m for m in onnx if "int8" in m["name"]), onnx[0] if onnx else quality)
        lines.append(
            f"| {name} | {teacher.get('name', '').removeprefix('teacher, ')} `{res.get('teacher')}` "
            f"| {_fmt(teacher.get('accuracy'), '.3f')} | `{res.get('student')}` "
            f"| {_fmt(quality.get('params_m'), '.0f')} "
            f"| {_pm(quality, 'accuracy')} | {_pm(quality, 'macro_f1')} | {_pm(quality, 'agreement')} "
            f"| {deployed['name'].removeprefix('student, ')} | {_fmt(deployed.get('latency_ms_b1'), '.1f')} "
            f"| {_fmt(deployed.get('latency_ms_b32'), '.1f')} | {_fmt(deployed.get('size_mb'), '.0f')} |"
        )
    threads = sorted({res["environment"]["cpu_threads"] for _, res in runs})
    note = f"\nAll latencies on the CPU with {', '.join(map(str, threads))} threads.\n" if threads else ""
    return head + "\n".join(lines) + "\n" + note


def cascade_curve(
    confidence: Sequence[float],
    student_pred: Sequence[Hashable],
    teacher_pred: Sequence[Hashable],
    gold: Sequence[Hashable],
    shares: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0),
) -> list[dict]:
    """Student-then-teacher cascade: for each share s, the s least confident texts (by the student's top
    probability) go to the teacher and the rest keep the student's answer. Returns, per share, the confidence
    threshold that achieves it, and the share actually sent, accuracy and macro-F1 of a cascade at that
    threshold (texts below it go; when confidences tie at the threshold, fewer than s go)."""
    n = len(gold)
    order = sorted(range(n), key=lambda i: confidence[i])  # least confident first
    out = []
    for share in shares:
        k = round(share * n)
        threshold = confidence[order[k]] if k < n else 1.0  # texts below this go to the teacher
        # as a Cascade at this threshold routes them: with ties (often at 1.0) fewer than k texts go
        routed = {i for i in range(n) if confidence[i] < threshold}
        pred = [teacher_pred[i] if i in routed else student_pred[i] for i in range(n)]
        out.append(
            {
                "share_to_teacher": len(routed) / n if n else 0.0,
                "threshold": threshold,
                "accuracy": accuracy(pred, gold),
                "macro_f1": macro_f1(pred, gold),
            }
        )
    return out


def cascade_table(
    curve: list[dict], student_ms: float | None, teacher_ms: float | None, teacher_name: str
) -> str:
    head = (
        "| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 "
        "| Mean latency per text (ms) |\n|---|---|---|---|---|\n"
    )
    rows = []
    for c in curve:
        latency = student_ms + c["share_to_teacher"] * teacher_ms if student_ms and teacher_ms else None
        rows.append(
            f"| {c['share_to_teacher']:.0%} | {c['threshold']:.3f} "
            f"| {c['accuracy']:.3f} | {c['macro_f1']:.3f} "
            f"| {_fmt(latency, '.1f')} |"
        )
    note = (
        "\nThe student answers every text and passes its least confident ones to the teacher. Thresholds are "
        "set on the test texts to hit each share; in deployment, set them on a validation split. "
        "Latency is the student's batch-1 latency plus the share times the batch-1 latency of the "
        f"{teacher_name}.\n"
    )
    return head + "\n".join(rows) + "\n" + note


def render(results: dict) -> str:
    """The whole report from results.json: the results table and, if there is one, the cascade table."""
    md = markdown_table(results)
    if results.get("cascade"):
        rows = results["models"]
        students = [m for m in rows if m["role"] == "student" and m.get("latency_ms_b1")]
        student = next((m for m in students if "int8" in m["name"]), students[-1] if students else None)
        # A cascade sends its unsure texts to a teacher on its fastest device, so the latency uses that one.
        teachers = [m for m in rows if m["role"] == "teacher" and m.get("latency_ms_b1")]
        fast = min(teachers, key=lambda m: m["latency_ms_b1"]) if teachers else None
        md += "\n### Cascade: student first, unsure texts to the teacher\n\n" + cascade_table(
            results["cascade"],
            student and student["latency_ms_b1"],
            fast and fast["latency_ms_b1"],
            f"{fast['name']} on {fast['device']}" if fast else "teacher",
        )
    return md
