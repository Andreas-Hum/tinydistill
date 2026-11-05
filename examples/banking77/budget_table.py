"""Write examples/banking77/budget.md from the saved results of budget.sh (and the full run of similar.sh)."""

import json
from pathlib import Path

here = Path(__file__).parent


def student(name: str) -> str:
    models = json.loads((here / f"{name}.json").read_text(encoding="utf-8"))["models"]
    m = next(m for m in models if m["name"] == "student, PyTorch")
    std = m.get("accuracy_std")
    return f"{m['accuracy']:.3f}" + (f" ± {std:.3f}" if std is not None else "")


rows = [
    "| Texts the teacher labels | At random (`--budget N`) | By the student's uncertainty (`--rounds 4`) |",
    "|---|---|---|",
]
for n in (1000, 2500, 5000):
    rows.append(f"| {n:,} of 10,003 | {student(f'budget{n}-r1')} | {student(f'budget{n}-r4')} |")
rows.append(f"| all 10,003 | {student('banking77-7b-similar16')} | |")
rows.append("")
rows.append(
    "Banking77, 3080 test texts; the teacher of similar.sh (16 most similar examples), its labels read from "
    "the score cache. Student accuracy (PyTorch), mean ± standard deviation over 3 training seeds."
)
(here / "budget.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
