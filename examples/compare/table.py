"""Write examples/compare/<task>.md, one table per task, from the runners' files in examples/compare/results/
and the saved tinydistill results in examples/ (examples/<task>/*.json, examples/<task>/students/*.json and
examples/teachers-2026/*/<task>-*.json). The tinydistill row is the saved run with the most accurate
student that uses at most 8 gold labels per class (a tie at three decimals goes to the better int8 model,
then the smaller), so a new run there is picked up when this script runs again."""

import json
from pathlib import Path

from common import EXAMPLES, GOLD_PER_CLASS, RESULTS, TASKS

from tinydistill.task import load_task

HERE = Path(__file__).parent
NAMES = {
    "banking77": "Banking77",
    "clinc150": "CLINC150 (plus, with out of scope)",
    "massive-da": "Danish MASSIVE",
    "financial-tweets": "Financial tweets",
}
# earlier runs with the default teacher, shown above the best run when they are not the best themselves
ALSO = {
    "banking77": ("banking77/banking77-7b-similar16.json",),
    "clinc150": ("clinc150/clinc150-7b-shortlist20-keep-oos.json",),
}
# reported in the papers, not rerun: (method, gold labels, also uses, accuracy, source)
PUBLISHED = {
    "banking77": [
        (
            "RoBERTa-Large + classifier, 10-shot",
            "770",
            "",
            0.8608,
            "Lin et al. 2023, Table 2 ([arXiv 2302.05096](https://arxiv.org/abs/2302.05096))",
        ),
        (
            "DNNC, 10-shot",
            "770",
            "pre-training on natural language inference datasets",
            0.8671,
            "Table 2 of [arXiv 2212.00301](https://arxiv.org/abs/2212.00301) and of arXiv 2302.05096",
        ),
        (
            "SetFit, MPNet-v2, 10-shot",
            "770, picked by a human expert",
            "",
            0.881,
            "Loukas et al. 2023, Table 4 ([arXiv 2311.06102](https://arxiv.org/abs/2311.06102)); micro-F1",
        ),
        (
            "RoBERTa-Large + ICDA-XL, 10-shot",
            "770",
            "128x as many texts written by OPT-66B, filtered by PVI on a validation set",
            0.8979,
            "Lin et al. 2023, Table 2 ([arXiv 2302.05096](https://arxiv.org/abs/2302.05096))",
        ),
    ],
}


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def pm(r: dict, key: str) -> str:
    std = r.get(f"{key}_std")
    return f"{r[key]:.3f}" + (f" ± {std:.3f}" if std else "")


def minutes(seconds: float) -> str:
    if seconds >= 5400:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 60:.0f} min" if seconds >= 90 else f"{seconds:.0f} s"


def mb(size: float) -> str:
    return f"{size / 1000:.1f} GB" if size >= 1000 else f"{size:.0f} MB"


def row(r: dict, name: str) -> dict | None:
    return next((m for m in r["models"] if m["name"] == name), None)


def tinydistill_runs(name: str, budget: int) -> list[tuple[Path, dict]]:
    """Saved tinydistill runs on this task and test set with at most 8 gold labels per class."""
    files = [*EXAMPLES.glob(f"{name}/*.json"), *EXAMPLES.glob(f"{name}/students/*.json")]
    files += EXAMPLES.glob(f"teachers-2026/*/{name}-*.json")
    out = []
    for f in sorted(files):
        r = read(f)
        # the same task under another task file, such as examples/clinc150/outside.yaml
        if not (isinstance(r, dict) and r.get("task", "").removesuffix("-outside") == name and "models" in r):
            continue
        t = r["training"]
        if r["data"]["test"] != TASKS[name][1] or t.get("gold_file") or (t.get("n_gold") or 0) > budget:
            continue
        if row(r, "student, PyTorch") and row(r, "student, ONNX int8"):
            out.append((f, r))
    return out


def teacher_facts(best: dict, runs: list[tuple[Path, dict]]) -> tuple[dict, float]:
    """The run's teacher row, and its labelling time. A run that read the teacher's scores from the cache, or
    skipped the teacher benchmark, takes them from a run of the same teacher and prompt on the same data."""
    teacher = next(m for m in best["models"] if m["role"] == "teacher")
    same = [
        (r, m)
        for _, r in runs
        for m in r["models"]
        if m["role"] == "teacher"
        and m["name"] == teacher["name"]
        and r["teacher"] == best["teacher"]
        and r["data"]["train"] == best["data"]["train"]
    ]
    if not teacher.get("latency_ms_b1"):
        timed = [m for _, m in same if m.get("latency_ms_b1")]
        teacher = timed[0] if timed else teacher
    seconds = best["training"].get("label_seconds") or max(
        (r["training"].get("label_seconds") or 0 for r, _ in same), default=0
    )
    members = best["teacher"].split(",")
    if not seconds and len(members) > 1:
        # an ensemble read from the cache its members' runs filled: their labelling is its labelling
        prompt = teacher["name"].split(", ", 1)[-1]
        seconds = sum(
            max(
                (
                    r["training"].get("label_seconds") or 0
                    for _, r in runs
                    if r["teacher"] == member
                    and r["data"]["train"] == best["data"]["train"]
                    and any(m["role"] == "teacher" and m["name"] == f"teacher, {prompt}" for m in r["models"])
                ),
                default=0,
            )
            for member in members
        )
    return teacher, seconds


def tinydistill_rows(path: Path, run: dict, runs: list[tuple[Path, dict]], every: list[dict]) -> list[str]:
    """Two table rows for a saved run: its teacher as the classifier, and its student."""
    rel = path.relative_to(EXAMPLES).as_posix()
    student, int8 = row(run, "student, PyTorch"), row(run, "student, ONNX int8")
    teacher, label_seconds = teacher_facts(run, runs)
    gpu = (run["environment"].get("gpu") or "GPU").removeprefix("NVIDIA ").removeprefix("GeForce ")
    t = run["training"]
    timed = next((r for r in every if r["method"] == "tinydistill" and r["results_file"] == rel), None)
    latency = f"{timed['latency_ms_b1']:.1f} ms" if timed else f"{int8['latency_ms_b1']:.1f} ms (saved)"
    on_gpu = str(teacher.get("device", "")).startswith("cuda")
    teacher_latency = (
        f"{teacher['latency_ms_b1']:.0f} ms ({gpu if on_gpu else 'CPU'})"
        if teacher.get("latency_ms_b1")
        else ""
    )
    prompt = teacher["name"].removeprefix("teacher").strip(" ,")
    shots = t.get("n_examples") or 0
    per_seed = t["train_seconds"] / (t.get("seeds") or 1)
    extra = " at most" if t.get("upper_bound") or t.get("gold_only") else ""
    return [
        "| The LLM as the classifier: "
        + ", ".join(f"`{m.split('/')[-1]}`" for m in run["teacher"].split(","))
        + f", {prompt} | {shots}{' (in the prompt)' if shots else ''} | | "
        f"{teacher['accuracy']:.3f} | {teacher['macro_f1']:.3f} | {teacher_latency} "
        f"| {mb(teacher['size_mb']) if teacher.get('size_mb') else ''} | none | yes, to run it |",
        f"| tinydistill: `{run['student'].split('/')[-1]}` student of that LLM, ONNX int8 "
        f"| {t.get('n_gold') or 0} | the LLM's labels on {run['data']['n_train']:,} unlabelled texts "
        f"| {pm(student, 'accuracy')} | {pm(student, 'macro_f1')} | {latency} | {mb(int8['size_mb'])} "
        f"| {minutes(label_seconds)} labelling +{extra} {minutes(per_seed)} training ({gpu}) "
        "| yes, to build it |",
    ]


def table(name: str) -> str | None:
    task = load_task(EXAMPLES / name / "task.yaml")
    budget = GOLD_PER_CLASS * len(task.labels)
    every = [read(f) for f in sorted(RESULTS.glob("*.json"))]
    ours = [r for r in every if r["task"] == name]
    runs = tinydistill_runs(name, budget)
    if not ours and not runs:
        return None
    n_gold = next((r["n_gold"] for r in ours if r.get("n_gold")), budget)
    threads = sorted({r["train_threads"] for r in ours if r.get("train_threads")})
    lines = [
        "| Method | Gold labels | Also uses | Accuracy | Macro-F1 | Latency per text, CPU | Size "
        "| Build time | GPU needed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    order = {"gold only": 0, "SetFit": 1, "GLiClass": 2}
    for r in sorted(
        (r for r in ours if r["method"] in order),
        key=lambda r: (order[r["method"]], r["model"], r.get("short", "")),
    ):
        short = r["model"].split("/")[-1]
        if r["method"] == "gold only":
            method = f"Gold only: `{short}` fine-tuned on the gold labels, ONNX int8"
            build = f"{minutes(r['train_seconds'])} (CPU)"
        elif r["method"] == "SetFit":
            method = f"SetFit, `{short}`"
            build = f"{minutes(r['train_seconds'])} (CPU)"
        else:
            method = f"GLiClass zero-shot, `{short}`"
            build = "none"
        uses = "label names" + (" and descriptions" if "descriptions" in r.get("short", "") else "")
        uses = uses if r["method"] == "GLiClass" else ""
        lines.append(
            f"| {method} | {r['n_gold'] or 0} | {uses} | {pm(r, 'accuracy')} | {pm(r, 'macro_f1')} "
            f"| {r['latency_ms_b1']:.1f} ms | {mb(r['size_mb'])} | {build} | no |"
        )
    notes = []
    if runs:
        # a tie at the precision shown goes to the better int8 model (the one deployed), then the smaller
        best_path, best = max(
            runs,
            key=lambda fr: (
                round(row(fr[1], "student, PyTorch")["accuracy"], 3),
                round(row(fr[1], "student, ONNX int8")["accuracy"], 3),
                -row(fr[1], "student, ONNX int8")["size_mb"],
                row(fr[1], "student, PyTorch")["accuracy"],
            ),
        )
        shown = [(p, r) for p, r in runs if p.relative_to(EXAMPLES).as_posix() in ALSO.get(name, ())]
        shown = [(p, r) for p, r in shown if p != best_path] + [(best_path, best)]
        for path, run in shown:
            lines += tinydistill_rows(path, run, runs, every)
        rel = best_path.relative_to(EXAMPLES).as_posix()
        timed = next((r for r in every if r["method"] == "tinydistill" and r["results_file"] == rel), None)
        int8 = timed or row(best, "student, ONNX int8")
        notes.append(
            f"The last tinydistill row is [`examples/{rel}`](https://github.com/Andreas-Hum/tinydistill/blob/"
            f"main/examples/{rel}), the most accurate of {len(runs)} saved runs on this test set that use at "
            f"most {GOLD_PER_CLASS} gold labels per class (picked on the test set, a tie at three decimals "
            "going to the better int8 model, then the smaller); the int8 model it times scores "
            f"{int8['accuracy']:.3f}."
        )
        if best["task"] != name:
            notes.append(
                'That run labels the out-of-scope requests as an `outside` label, "none of these" '
                "(examples/clinc150/outside.yaml); the test texts and their gold labels are the same."
            )
        if len(shown) > 1:
            notes.append(
                "The LLM and tinydistill rows before it are an earlier run, kept for reference: "
                + ", ".join(f"`examples/{p.relative_to(EXAMPLES).as_posix()}`" for p, _ in shown[:-1])
                + "."
            )
    n_test = next((r["n_test"] for r in ours), runs[0][1]["data"]["n_test"] if runs else 0)
    head = (
        f"{NAMES[name]}, {n_test:,} test texts. Every method that trains gets the same {n_gold} "
        f"gold-labelled texts ({GOLD_PER_CLASS} per class, tinydistill's own pick with seed 0); mean ± "
        "standard deviation over 3 training seeds. Latency: batch 1, 2 threads, median over three passes "
        "through the same 100 test texts, each library's own inference (tinydistill's Predictor for the ONNX "
        "models); the LLM runs on a GPU. The gold-only and tinydistill accuracies are the PyTorch models'; "
        'their latency and size are those of the int8 export ("saved": timed when the run was made, '
        "not re-timed with the others). "
        "Build time is per model"
        + (f"; the CPU runs trained with {' or '.join(map(str, threads))} threads." if threads else ".")
    )
    out = "\n".join(lines) + "\n\n" + " ".join([head, *notes]) + "\n"
    if name in PUBLISHED:
        out += (
            "\nReported in the papers, not rerun. The protocol differs from the table above: 10 gold labels "
            f"per class (770) against our {GOLD_PER_CLASS} ({n_gold}), plus what the third column says, and "
            "larger classifiers (RoBERTa-Large, 355M parameters; MPNet, 110M). The test set is the same "
            "3,080 texts. Lin et al. sample the 10 per class at random, average 5 seeds and train "
            "RoBERTa-Large for 40 epochs; Loukas et al. train on 10 per class that a human expert picked "
            "as representative.\n\n"
            "| Method | Gold labels | Also uses | Accuracy | Source |\n|---|---|---|---|---|\n"
        )
        out += "".join(f"| {m} | {g} | {u} | {a:.3f} | {s} |\n" for m, g, u, a, s in PUBLISHED[name])
    return out


for name in TASKS:
    text = table(name)
    if text:
        (HERE / f"{name}.md").write_text(text, encoding="utf-8")
        print(f"wrote examples/compare/{name}.md")
