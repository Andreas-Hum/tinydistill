"""Fill every `<!-- results: <file> -->` ... `<!-- /results -->` block in the README with that results file
(relative to examples/). Blocks whose file does not exist yet are left as they are."""

import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
readme = root / "README.md"
block = re.compile(r"(<!-- results: (?P<path>\S+) -->\n).*?(<!-- /results -->)", re.S)


def fill(m: re.Match) -> str:
    path = root / "examples" / m["path"]
    if not path.exists():
        return m.group(0)
    return f"{m.group(1)}{path.read_text(encoding='utf-8').strip()}\n{m.group(3)}"


readme.write_text(block.sub(fill, readme.read_text(encoding="utf-8")), encoding="utf-8")
print("README results updated")
