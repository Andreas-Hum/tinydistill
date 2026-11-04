"""Charts of a run, as matplotlib figures (``pip install "tinydistill[plot]"``). Each function takes a run
directory and returns a figure, so in a notebook it shows inline; ``save_all`` writes every chart the run has
data for to ``RUN/plots/*.png`` (that is what `tinydistill plot RUN` does).

The student is blue and the teacher orange throughout, as in the web page."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

STUDENT, TEACHER, REFERENCE = "#2a78d6", "#eb6834", "#8a8984"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _plt():
    try:
        import matplotlib

        if not _in_notebook():
            matplotlib.use("Agg")  # files only, no window
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit('plots need matplotlib: pip install "tinydistill[plot]"') from e
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "legend.frameon": False,
            "font.size": 10,
        }
    )
    return plt


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def _results(run: Path) -> dict:
    f = Path(run) / "results.json"
    if not f.exists():
        raise FileNotFoundError(f"{f} is missing: run `tinydistill eval` first")
    return json.loads(f.read_text(encoding="utf-8"))


def _name(label: str) -> str:
    return label.replace("_", " ")


def training(run: str | Path):
    """Training loss per epoch, and the held-out loss and the kept epoch when `--val-share` was used."""
    plt = _plt()
    f = Path(run) / "student" / "history.json"
    if not f.exists():
        raise FileNotFoundError(f"{f} is missing: train with tinydistill 0.3.3 or later")
    h = json.loads(f.read_text(encoding="utf-8"))
    epochs = [e["epoch"] for e in h["epochs"]]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    held = [e.get("held_out_loss") for e in h["epochs"]]
    has_held = any(v is not None for v in held)
    ax.plot(  # both lines are the student; the held-out loss decides, so it gets the student's colour
        epochs,
        [e["loss"] for e in h["epochs"]],
        color=REFERENCE if has_held else STUDENT,
        lw=2,
        marker="o",
        ms=5,
        label="training texts",
    )
    if has_held:
        ax.plot(
            epochs, held, color=STUDENT, lw=2, marker="o", ms=5, label=f"{h['held_out_texts']} held-out texts"
        )
        ax.axvline(h["kept_epoch"], color=REFERENCE, lw=1, ls="--")
        ax.annotate(
            f" kept: epoch {h['kept_epoch']}", (h["kept_epoch"], ax.get_ylim()[1]), va="top", color=MUTED
        )
    ax.set(xlabel="epoch", ylabel="loss (lower is better)", title="Training")
    ax.set_xticks(epochs)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def per_label(run: str | Path, limit: int | None = None):
    """F1 of every label for the teacher and the student, the teacher's weakest labels at the top. With
    ``limit``, only that many of the labels where the two differ most."""
    from .analyse import per_label as scores
    from .analyse import predictions

    plt = _plt()
    p = predictions(Path(run))
    names = p["task"].labels
    t = [s["f1"] for s in scores(p["teacher"], p["gold"], len(names))]
    s = [x["f1"] for x in scores(p["student"], p["gold"], len(names))]
    order = sorted(range(len(names)), key=lambda c: t[c])
    if limit:
        order = sorted(sorted(order, key=lambda c: -abs(s[c] - t[c]))[:limit], key=lambda c: t[c])
    fig, ax = plt.subplots(figsize=(7, 0.24 * len(order) + 1.4))
    y = range(len(order))
    ax.hlines(y, [min(t[c], s[c]) for c in order], [max(t[c], s[c]) for c in order], color=GRID, lw=2)
    ax.scatter([t[c] for c in order], y, color=TEACHER, s=36, zorder=3, label="teacher")
    ax.scatter([s[c] for c in order], y, color=STUDENT, s=36, zorder=3, label="student")
    ax.set_yticks(list(y), [_name(names[c]) for c in order], fontsize=8)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(0, 1.02)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="F1 on the test texts", title="Per label: teacher and student")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=2, borderaxespad=0.2)
    fig.tight_layout()
    return fig


def confusions(run: str | Path, who: str = "student", labels: int = 12):
    """Confusion matrix of the student (or ``who="teacher"``) over the ``labels`` labels involved in the most
    mistakes: rows are the gold labels, columns the predicted ones."""
    from .analyse import predictions

    plt = _plt()
    p = predictions(Path(run))
    names, gold, pred = p["task"].labels, p["gold"], p[who]
    errors: Counter[int] = Counter()
    for g, x in zip(gold, pred, strict=True):
        if g != x:
            errors[g] += 1
            errors[x] += 1
    keep = sorted(c for c, _ in errors.most_common(labels)) or list(range(min(labels, len(names))))
    index = {c: i for i, c in enumerate(keep)}
    m = [[0] * len(keep) for _ in keep]
    for g, x in zip(gold, pred, strict=True):
        if g in index and x in index:
            m[index[g]][index[x]] += 1
    fig, ax = plt.subplots(figsize=(0.5 * len(keep) + 3, 0.45 * len(keep) + 2))
    ax.imshow(m, cmap="Blues")
    top = max(max(r) for r in m) or 1
    for i, r in enumerate(m):
        for j, v in enumerate(r):
            if v:
                ax.text(
                    j, i, v, ha="center", va="center", fontsize=8, color="#ffffff" if v > top / 2 else INK
                )
    ticks = [_name(names[c]) for c in keep]
    ax.set_xticks(range(len(keep)), ticks, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(keep)), ticks, fontsize=8)
    ax.grid(False)
    ax.set(xlabel="predicted", ylabel="gold", title=f"Most confused labels ({who})")
    fig.tight_layout()
    return fig


def speed_accuracy(run: str | Path):
    """Accuracy against latency per text (batch 1, log scale) for every timed model, with the upper bound and
    the gold-only baseline as reference lines when the run has them."""
    plt = _plt()
    rows = _results(Path(run))["models"]
    timed = [r for r in rows if r.get("latency_ms_b1") and r.get("accuracy") is not None]
    if not timed:
        raise ValueError("no timed model with an accuracy in results.json")
    fig, ax = plt.subplots(figsize=(7, 4))
    students = [r for r in timed if not r["role"].startswith("teacher")]
    deployed = min(students, key=lambda r: r["latency_ms_b1"]) if students else None  # e.g. ONNX int8
    for r in timed:
        color = TEACHER if r["role"].startswith("teacher") else STUDENT
        ax.scatter(r["latency_ms_b1"], r["accuracy"], color=color, s=60, zorder=3, edgecolor=SURFACE, lw=1.5)
        if color == STUDENT and r is not deployed:
            continue  # label only the teacher and the fastest student, so labels never collide
        label = r["name"].replace("teacher, ", "teacher ").replace("student, ", "student ")
        ax.annotate(
            f"{label} ({r['device']})",
            (r["latency_ms_b1"], r["accuracy"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color=MUTED,
        )
    for r in rows:
        if r["role"] in ("upper bound", "baseline") and r.get("accuracy") is not None:
            ls = "--" if r["role"] == "upper bound" else ":"
            ax.axhline(r["accuracy"], color=REFERENCE, lw=1, ls=ls)
            ax.annotate(
                r["role"],
                (1, r["accuracy"]),
                xycoords=("axes fraction", "data"),
                ha="right",
                va="bottom",
                fontsize=8,
                color=MUTED,
            )
    from matplotlib.ticker import FuncFormatter

    ax.set_xscale("log")
    ax.margins(x=0.3)  # room for the labels on both sides
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if f"{v:g}"[0] in "25" else ""))
    accs = [r["accuracy"] for r in rows if r.get("accuracy") is not None]
    ax.set_ylim(max(0.0, min(accs) - 0.1), min(1.0, max(accs) + 0.05))  # never zoom in on a tiny gap
    ax.set(
        xlabel="latency per text, batch 1 (ms, log scale)", ylabel="test accuracy", title="Accuracy and speed"
    )
    roles = {r["role"].startswith("teacher") for r in timed}
    for is_teacher, color, label in ((True, TEACHER, "teacher"), (False, STUDENT, "student")):
        if is_teacher in roles:
            ax.scatter([], [], color=color, label=label)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def cascade(run: str | Path):
    """Accuracy as the student passes its least confident texts to the teacher."""
    plt = _plt()
    points = _results(Path(run)).get("cascade")
    if not points:
        raise ValueError("results.json has no cascade (single-label runs with a gold-labelled test set)")
    x = [100 * c["share_to_teacher"] for c in points]
    y = [c["accuracy"] for c in points]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(x, y, color=STUDENT, lw=2, marker="o", ms=6)
    for i, text, ha in ((0, "student only", "left"), (-1, "teacher only", "right")):
        ax.annotate(
            text, (x[i], y[i]), textcoords="offset points", xytext=(0, -16), ha=ha, fontsize=8, color=MUTED
        )
    ax.margins(y=0.15)
    ax.set(
        xlabel="texts sent to the teacher (%)",
        ylabel="test accuracy",
        title="Cascade: unsure texts go to the teacher",
    )
    fig.tight_layout()
    return fig


def soft_labels(run: str | Path, n: int = 3, k: int = 4):
    """For ``n`` test texts with different gold labels: the teacher's probabilities (the soft labels the
    student learned from) next to the student's, over the ``k`` most likely labels."""
    import math

    from .export import Predictor
    from .task import load_task
    from .web import _samples

    plt = _plt()
    run = Path(run)
    task = load_task(run / "task.yaml")
    if task.multi_label:
        raise ValueError("soft_labels is for single-label tasks")
    rows = [
        json.loads(line) for line in (run / "test.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    rows = _samples([r for r in rows if r.get("scores")], n)
    if not rows:
        raise ValueError("the run has no test texts with teacher scores")
    student = Predictor(run / "onnx").predict_proba([r["text"] for r in rows])
    fig, axes = plt.subplots(
        len(rows), 1, figsize=(7, 1.55 * len(rows) * max(1, k / 4) + 0.6), squeeze=False, sharex=True
    )
    for ax, r, q in zip(axes[:, 0], rows, student, strict=True):
        m = max(r["scores"])
        e = [math.exp(x - m) for x in r["scores"]]
        p = [x / sum(e) for x in e]
        top = sorted(range(len(p)), key=lambda c: -(p[c] + q[c]))[:k]
        y = range(len(top))
        ax.barh([i - 0.2 for i in y], [p[c] for c in top], height=0.38, color=TEACHER, label="teacher")
        ax.barh([i + 0.2 for i in y], [q[c] for c in top], height=0.38, color=STUDENT, label="student")
        gold = r.get("gold")
        ax.set_yticks(
            list(y), [_name(task.labels[c]) + ("  (gold)" if c == gold else "") for c in top], fontsize=8
        )
        ax.set_ylim(len(top) - 0.5, -0.5)
        ax.set_xlim(0, 1)
        ax.grid(axis="y", visible=False)
        text = r["text"] if len(r["text"]) <= 90 else r["text"][:87] + "..."
        ax.set_title(f"\u201c{text}\u201d", fontsize=9, fontweight="normal", color=MUTED)
    axes[0, 0].legend(loc="lower right", bbox_to_anchor=(1, 1.18), ncols=2, borderaxespad=0)
    axes[-1, 0].set_xlabel("probability")
    fig.suptitle("Soft labels: teacher and student", x=0.02, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    return fig


CHARTS = {
    "training": training,
    "speed_accuracy": speed_accuracy,
    "per_label": per_label,
    "confusions": confusions,
    "cascade": cascade,
    "soft_labels": soft_labels,
}


def save_all(run: str | Path, out: str | Path | None = None) -> list[Path]:
    """Write every chart the run has data for to ``out`` (default ``RUN/plots``); returns the files."""
    plt = _plt()
    out = Path(out) if out else Path(run) / "plots"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, chart in CHARTS.items():
        try:
            fig = chart(run)
        except (FileNotFoundError, ValueError, SystemExit):
            continue  # this run has no data for the chart (e.g. multi-label, or no cascade)
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)
        written.append(out / f"{name}.png")
    return written
