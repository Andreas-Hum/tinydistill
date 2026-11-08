"""The probe-layer sweep's table: for each run of probe-layers.sh, the probe's test accuracy (its top label
against the gold one) and macro-F1, read from the run's test.jsonl. Writes <out>.json and <out>.md.

    python examples/teachers-2026/probe_layers.py OUT RUN [RUN ...]
"""

import json
import sys
from pathlib import Path


def scores(run: Path) -> dict:
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (run / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    pred = [max(range(len(r["scores"])), key=r["scores"].__getitem__) for r in rows]
    gold = [r["gold"] for r in rows]
    f1 = []
    for c in set(gold):
        tp = sum(p == g == c for p, g in zip(pred, gold, strict=True))
        n_pred, n_gold = pred.count(c), gold.count(c)
        f1.append(2 * tp / (n_pred + n_gold) if n_pred + n_gold else 0.0)
    return {
        "run": run.name,
        "teacher": meta["teacher"],
        "layer": meta["probe"]["layer"],
        "prompt": f"{meta['few_shot_similar']} similar examples" if meta["few_shot_similar"] else "zero-shot",
        "label_seconds": meta["label_seconds"],
        "n_test": len(rows),
        "accuracy": sum(p == g for p, g in zip(pred, gold, strict=True)) / len(rows),
        "macro_f1": sum(f1) / len(f1),
    }


def main() -> None:
    out, runs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    rows = [scores(r) for r in runs]
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    lines = [
        "| Teacher | Prompt | Probe reads | Teacher accuracy | Macro-F1 | Labelling |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        where = "LM head input (default)" if r["layer"] is None else f"layer {r['layer']}"
        lines.append(
            f"| `{r['teacher']}` | {r['prompt']} | {where} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | "
            + (f"{r['label_seconds']:.0f} s |" if r["label_seconds"] else "from the cache |")
        )
    n = rows[0]["n_test"]
    lines += ["", f"Test accuracy of the probe alone (`label --probe-only`) on {n:,} test texts."]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
