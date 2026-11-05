"""Build a `tinydistill web` page and its explainer from a tiny random model, and an explainer for a
multi-label one, for check_page.mjs and check_explainer.mjs (the CI browser tests).

python tests/browser/build_tiny_site.py site
python tests/browser/build_tiny_site.py site-multi --multi   # multi-label, no layer or attention outputs
"""

import json
import shutil
import sys
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tinydistill.explainer import build_explainer
from tinydistill.export import export_onnx
from tinydistill.web import build_site

ENCODER = "hf-internal-testing/tiny-random-BertForSequenceClassification"

out = Path(sys.argv[1] if len(sys.argv) > 1 else "site")
multi = "--multi" in sys.argv
run = out.parent / f"{out.name}-run"
model = AutoModelForSequenceClassification.from_pretrained(
    ENCODER,
    num_labels=2,
    id2label={0: "negative", 1: "positive"},
    ignore_mismatched_sizes=True,
    problem_type="multi_label_classification" if multi else None,
)
model.save_pretrained(run / "student")
AutoTokenizer.from_pretrained(ENCODER).save_pretrained(run / "student")
export_onnx(run / "student", run / "onnx", int8=True)
if multi:  # without the PyTorch student the page has no layer or attention outputs
    shutil.rmtree(run / "student")
rows = [
    {"text": "what a wonderful day, I loved every minute", "gold": 1, "scores": [-3.0, -0.05]},
    {"text": "this was slow, dull and far too long", "gold": 0, "scores": [-0.1, -2.4]},
]
if multi:
    rows = [
        {"text": "what a wonderful day, I loved every minute", "gold": [1], "scores": [-3.0, 2.0]},
        {"text": "this was slow, dull and far too long", "gold": [0], "scores": [1.5, -2.4]},
        {"text": "good and bad in equal parts", "gold": [0, 1], "scores": [0.4, 0.3]},
    ]
(run / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
results = {
    "teacher": "a teacher",
    "student": ENCODER,
    "data": {"n_test": len(rows)},
    "training": {"ce_weight": 1.0},
    "models": [
        {
            "role": "teacher",
            "name": "teacher, zero-shot",
            "device": "cuda",
            "accuracy": 1.0,
            "macro_f1": 1.0,
            "size_mb": 3087.4,
            "latency_ms_b1": 60.3,
        },
        {"role": "student", "name": "student, ONNX int8", "accuracy": 0.5, "macro_f1": 0.4, "size_mb": 0.5},
    ],
}
if not multi:
    results["cascade"] = [
        {"share_to_teacher": 0.0, "threshold": 0.4, "accuracy": 0.5},
        {"share_to_teacher": 0.5, "threshold": 0.6, "accuracy": 0.75},
        {"share_to_teacher": 1.0, "threshold": 1.0, "accuracy": 1.0},
    ]
(run / "results.json").write_text(json.dumps(results), encoding="utf-8")
# as `tinydistill web` does: the explainer is the main page and the plain page goes to simple/; the
# multi-label model has no PyTorch student (removed above), so its plain page is its only page
if not multi:
    print(build_explainer(run / "onnx", out, title="Browser test"))
simple = out if multi else out / "simple"
print(build_site(run / "onnx", simple, title="Browser test", explainer=not multi))
if multi:
    raise SystemExit

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
