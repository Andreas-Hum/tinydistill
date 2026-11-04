"""The CLINC150 runs side by side: labelling time, the teacher's accuracy on in-scope and out-of-scope
requests, how often the gold label was shortlisted, and the student. Reads runs/clinc150-*; writes
examples/clinc150/shortlist.json and shortlist.md."""

import json
from pathlib import Path

from tinydistill.task import load_task

HERE = Path(__file__).parent
RUNS = {
    "clinc150-7b-full": "every label (151)",
    "clinc150-7b-shortlist20": "`--shortlist 20`",
    "clinc150-7b-shortlist20-keep-oos": "`--shortlist 20 --shortlist-keep oos`",
}
task = load_task(HERE / "task.yaml")
oos, n = task.labels.index("oos"), len(task.labels)
summary = []
for run, scored in RUNS.items():
    d = Path("runs") / run
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (d / "test.jsonl").read_text(encoding="utf-8").splitlines() if line]
    teacher = next(m for m in res["models"] if m["role"] == "teacher")
    student = next(m for m in res["models"] if m["name"] == "student, ONNX int8")

    def accuracy(sel: list[dict]) -> float:
        return sum(max(range(n), key=r["scores"].__getitem__) == r["gold"] for r in sel) / len(sel)

    summary.append(
        {
            "run": run,
            "scored": scored,
            "label_minutes": meta["label_seconds"] / 60,
            "texts_labelled": meta["label_texts_scored"],
            "teacher_accuracy": teacher["accuracy"],
            "teacher_in_scope": accuracy([r for r in rows if r["gold"] != oos]),
            "teacher_out_of_scope": accuracy([r for r in rows if r["gold"] == oos]),
            "shortlist_recall": res["training"].get("shortlist_recall"),
            "teacher_latency_ms": teacher.get("latency_ms_b1"),
            "student_accuracy": student["accuracy"],
        }
    )
(HERE / "shortlist.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
lines = [
    "| Teacher scores | Labelling time | Teacher accuracy | In scope | Out of scope | Gold label scored "
    "| Teacher latency (GPU) | Student accuracy (int8, CPU) |",
    "|---|---|---|---|---|---|---|---|",
]
for s in summary:
    recall = f"{s['shortlist_recall']:.1%}" if s["shortlist_recall"] is not None else "100%"
    lines.append(
        f"| {s['scored']} | {s['label_minutes']:.0f} min | {s['teacher_accuracy']:.3f} "
        f"| {s['teacher_in_scope']:.3f} | {s['teacher_out_of_scope']:.3f} | {recall} "
        f"| {s['teacher_latency_ms']:.0f} ms | {s['student_accuracy']:.3f} |"
    )
note = (
    "\nCLINC150 (plus), 5,500 test texts of which 1,000 are out of scope; "
    "`Qwen/Qwen2.5-7B-Instruct` in 4 bit, zero-shot, labelling about "
    f"{summary[0]['texts_labelled']:,} training and test texts on one GPU. "
    "Gold label scored: the share of test texts whose gold label the teacher scored at all.\n"
)
(HERE / "shortlist.md").write_text("\n".join(lines) + "\n" + note, encoding="utf-8")
print((HERE / "shortlist.md").read_text(encoding="utf-8"))
