"""Build a `tinydistill web` page from a tiny random model, for check_page.mjs (the CI browser test).

python tests/browser/build_tiny_site.py site
"""

import json
import sys
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tinydistill.export import export_onnx
from tinydistill.web import build_site

ENCODER = "hf-internal-testing/tiny-random-BertForSequenceClassification"

out = Path(sys.argv[1] if len(sys.argv) > 1 else "site")
run = out.parent / f"{out.name}-run"
model = AutoModelForSequenceClassification.from_pretrained(
    ENCODER, num_labels=2, id2label={0: "negative", 1: "positive"}, ignore_mismatched_sizes=True
)
model.save_pretrained(run / "student")
AutoTokenizer.from_pretrained(ENCODER).save_pretrained(run / "student")
export_onnx(run / "student", run / "onnx", int8=True)
rows = [
    {"text": "what a wonderful day, I loved every minute", "gold": 1, "scores": [-3.0, -0.05]},
    {"text": "this was slow, dull and far too long", "gold": 0, "scores": [-0.1, -2.4]},
]
(run / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
(run / "results.json").write_text(
    json.dumps(
        {
            "teacher": "a teacher",
            "student": ENCODER,
            "data": {"n_test": 2},
            "models": [
                {"role": "teacher", "name": "teacher, zero-shot"},
                {"role": "student", "name": "student, ONNX int8", "accuracy": 0.5},
            ],
            "cascade": [
                {"share_to_teacher": 0.0, "threshold": 0.4, "accuracy": 0.5},
                {"share_to_teacher": 0.5, "threshold": 0.6, "accuracy": 0.75},
                {"share_to_teacher": 1.0, "threshold": 1.0, "accuracy": 1.0},
            ],
        }
    ),
    encoding="utf-8",
)
print(build_site(run / "onnx", out, title="Browser test"))
