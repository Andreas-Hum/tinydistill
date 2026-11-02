"""Copy examples/banking77/results.md into the README between the results markers."""

from pathlib import Path

root = Path(__file__).resolve().parents[2]
readme = root / "README.md"
table = (root / "examples/banking77/results.md").read_text().strip()
text = readme.read_text()
start, end = "<!-- results:start -->", "<!-- results:end -->"
head, rest = text.split(start)
_, tail = rest.split(end)
readme.write_text(f"{head}{start}\n{table}\n{end}{tail}")
print("README results updated")
