"""Write examples/banking77/warmup.md from the saved results of warmup.sh, budget.sh and similar.sh."""

import json
from pathlib import Path

here = Path(__file__).parent


def student(name: str, model: str = "student, PyTorch") -> str:
    models = json.loads((here / f"{name}.json").read_text(encoding="utf-8"))["models"]
    m = next(m for m in models if m["name"] == model)
    std = m.get("accuracy_std")
    return f"{m['accuracy']:.3f}" + (f" ± {std:.3f}" if std is not None else "")


rows = [
    "| Texts the teacher labels | Without warm-up | With `--warmup-pairs 20000` |",
    "|---|---|---|",
]
for n in (1000, 2500):
    rows.append(
        f"| {n:,} of 10,003, at random | {student(f'budget{n}-r1')} | {student(f'budget{n}-warmup')} |"
    )
full, warm = "banking77-7b-similar16", "banking77-7b-similar16-warmup"
rows.append(f"| all 10,003 | {student(full)} | {student(warm)} |")
rows.append(
    f"| all 10,003, deployed (ONNX int8) | {student(full, 'student, ONNX int8')} "
    f"| {student(warm, 'student, ONNX int8')} |"
)
rows.append("")
rows.append(
    "Banking77, 3080 test texts; the teacher of similar.sh (16 most similar examples), its labels read from "
    "the score cache. Student accuracy (PyTorch unless marked), mean ± standard deviation over 3 training "
    "seeds."
)
(here / "warmup.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
