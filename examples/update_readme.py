"""Fill every `<!-- results: <file> -->` ... `<!-- /results -->` block in README.md and RESULTS.md with that
results file (relative to examples/). Blocks whose file does not exist yet are left as they are.

It first writes examples/headline.md, the README's summary table, from the saved results files."""

import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
examples = root / "examples"
block = re.compile(r"(<!-- results: (?P<path>\S+) -->\n).*?(<!-- /results -->)", re.S)


def row(results: dict, name: str) -> dict:
    return next(m for m in results["models"] if m["name"] == name)


def headline() -> str | None:
    """Banking77, the live demo's run: the teacher on the GPU against its int8 ONNX student on the CPU."""
    files = {
        "teacher": examples / "banking77" / "banking77-7b-similar16.json",
        "minilm": examples / "banking77" / "banking77-7b-similar16.json",
    }
    if not all(f.exists() for f in files.values()):
        return None
    res = {k: json.loads(f.read_text(encoding="utf-8")) for k, f in files.items()}
    teacher = next(m for m in res["teacher"]["models"] if m["role"] == "teacher")
    env = res["teacher"]["environment"]
    gpu = (env.get("gpu") or "GPU").removeprefix("NVIDIA ").removeprefix("GeForce ")
    lines = [
        "| Model | Runs on | Test accuracy | Latency per text | Size | Speed-up |",
        "|---|---|---|---|---|---|",
        f"| Teacher: `{res['teacher']['teacher']}`, 4-bit, 16 most similar examples in the prompt | {gpu} "
        f"| {teacher['accuracy']:.3f} | {teacher['latency_ms_b1']:.0f} ms "
        f"| {teacher['size_mb'] / 1000:.1f} GB | 1x |",
    ]
    for key, label in (("minilm", "Student: MiniLM-L6, int8 ONNX"),):
        s = row(res[key], "student, ONNX int8")
        lines.append(
            f"| {label} | CPU, {res[key]['environment']['cpu_threads']} threads | {s['accuracy']:.3f} "
            f"| {s['latency_ms_b1']:.1f} ms | {s['size_mb']:.0f} MB "
            f"| {teacher['latency_ms_b1'] / s['latency_ms_b1']:.0f}x |"
        )
    return "\n".join(lines) + "\n"


def fill(m: re.Match) -> str:
    path = examples / m["path"]
    if not path.exists():
        return m.group(0)
    return f"{m.group(1)}{path.read_text(encoding='utf-8').strip()}\n{m.group(3)}"


table = headline()
if table:
    (examples / "headline.md").write_text(table, encoding="utf-8")
for name in ("README.md", "RESULTS.md"):
    page = root / name
    if page.exists():
        page.write_text(block.sub(fill, page.read_text(encoding="utf-8")), encoding="utf-8")
print("README and RESULTS updated")
