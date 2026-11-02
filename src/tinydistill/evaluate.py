"""Quality metrics, latency benchmarks and the results table."""

from __future__ import annotations

import os
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch


def accuracy(pred: Sequence[int], gold: Sequence[int]) -> float:
    return sum(p == g for p, g in zip(pred, gold, strict=True)) / len(gold) if gold else float("nan")


def macro_f1(pred: Sequence[int], gold: Sequence[int]) -> float:
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


def bench(fn: Callable[[list[str]], object], texts: list[str], reps1: int, reps32: int = 0) -> dict:
    """Latency at batch 1 and (if reps32 > 0) batch 32. Throughput comes from batch 32 when measured,
    otherwise from batch 1 (for models that process one text at a time)."""
    out = {"latency_ms_b1": time_calls(fn, texts, 1, reps1)}
    if reps32:
        out["latency_ms_b32"] = time_calls(fn, texts, 32, reps32)
        out["throughput_per_s"] = 32 / (out["latency_ms_b32"] / 1000)
    else:
        out["throughput_per_s"] = 1000 / out["latency_ms_b1"]
    return out


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


def markdown_table(results: dict) -> str:
    rows = results["models"]
    ref = next((r for r in rows if r["role"] == "teacher" and r.get("latency_ms_b1")), None)
    head = (
        "| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) "
        "| Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        speed = ref["latency_ms_b1"] / r["latency_ms_b1"] if ref and r.get("latency_ms_b1") else None
        lines.append(
            f"| {r['name']} | {r['device']} "
            f"| {_pm(r, 'accuracy')} | {_pm(r, 'macro_f1')} "
            f"| {_pm(r, 'agreement')} | {_fmt(r.get('latency_ms_b1'), '.1f')} "
            f"| {_fmt(r.get('latency_ms_b32'), '.1f')} | {_fmt(r.get('throughput_per_s'), '.1f')} "
            f"| {_fmt(r.get('size_mb'), '.0f')} | {_fmt(speed, '.1f') + 'x' if speed else 'n/a'} |"
        )
    ds = results["data"]
    tr = results.get("training") or {}
    k = tr.get("gold_per_class") or 0
    trained = (
        f"The student learns from the teacher's soft labels on all {ds['n_train']} training texts, "
        f"plus cross-entropy on {k} gold labels per class "
        f"({tr.get('n_gold')} texts, weight {tr.get('ce_weight')})."
        if k
        else f"The student learns only from the teacher's soft labels on {ds['n_train']} "
        "unlabelled training texts."
    )
    if tr.get("few_shot"):
        trained = (
            f"The teacher's prompt shows {tr['few_shot']} gold training examples per class "
            f"({tr.get('n_examples')} texts). " + trained
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
