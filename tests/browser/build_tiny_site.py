"""Build a `tinydistill web` page and its explainer from a tiny random model, and an explainer for a
multi-label one, for check_page.mjs and check_explainer.mjs (the CI browser tests).

python tests/browser/build_tiny_site.py site
"""

import json
import sys
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tinydistill.explainer import build_explainer
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
print(build_explainer(run / "onnx", out / "explainer", title="Browser test"))  # as `tinydistill web` does

# a multi-label model for the explainer too, at out/explainer-multi
multi = out.parent / f"{out.name}-multi"
model = AutoModelForSequenceClassification.from_pretrained(
    ENCODER,
    num_labels=3,
    id2label={0: "sports", 1: "politics", 2: "money"},
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True,
)
model.save_pretrained(multi / "student")
AutoTokenizer.from_pretrained(ENCODER).save_pretrained(multi / "student")
export_onnx(multi / "student", multi / "onnx", int8=True)
row = {"text": "the minister bought the football club", "gold": [0, 1, 2], "scores": [1.5, 2.0, 0.3]}
(multi / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
print(build_explainer(multi / "onnx", out / "explainer-multi", title="Browser test, multi-label"))
