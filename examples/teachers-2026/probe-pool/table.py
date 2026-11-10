"""The markdown table of pool.py's results: per run, the probe teacher's test accuracy for each kind of
state.

    python examples/teachers-2026/probe-pool/table.py examples/teachers-2026/probe-pool/similar16.json \\
        > examples/teachers-2026/probe-pool/similar16.md
"""

import json
import sys
from pathlib import Path

COLUMNS = [
    ("last", "last token, LM head input (default)"),
    ("mean", "mean, LM head input"),
    ("last_layer", "last token, layer"),
    ("mean_layer", "mean, layer"),
]

rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("| Run | Teacher | Layer | " + " | ".join(name for _, name in COLUMNS) + " |")
print("|---|---|---|" + "---|" * len(COLUMNS))
for r in rows:
    cells = [f"{r[k]['accuracy']:.3f}" for k, _ in COLUMNS]
    print(f"| {r['run']} | `{r['teacher']}` | {r['layer']} of {r['n_layers']} | " + " | ".join(cells) + " |")
print()
print(
    "Test accuracy of the probe alone, fit on the same gold texts, reading the teacher's state at the "
    "text's last token or the mean of its states over the text's own tokens (the text and the prompt after "
    "it, not the examples before it), at the LM head's input or at the output of the layer given."
)
