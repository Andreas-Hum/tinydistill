"""Build a `tinydistill web` page and its explainer from a tiny random model, and an explainer for a
multi-label one, for check_page.mjs and check_explainer.mjs (the CI browser tests).

python tests/browser/build_tiny_site.py site
python tests/browser/build_tiny_site.py site-multi --multi   # multi-label, no layer or attention outputs

The first also writes site/hub/, a local copy of the Hugging Face Hub with three repositories as `tinydistill
push` uploads them (test/multi, test/plain without an explainer, and test/broken, whose model files are
missing), which the pages list in their model picker (`web --models`), and site/remote/, the pages of
`web hf:test/multi --remote`. The checks point the pages at it with HUB=http://.../hub/.
"""

import json
import shutil
import sys
from pathlib import Path

import huggingface_hub
from huggingface_hub.errors import EntryNotFoundError
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tinydistill.cli import main
from tinydistill.explainer import build_explainer
from tinydistill.export import export_onnx
from tinydistill.hub import repo_files
from tinydistill.web import build_site

ENCODER = "hf-internal-testing/tiny-random-BertForSequenceClassification"

out = Path(sys.argv[1] if len(sys.argv) > 1 else "site")
multi = "--multi" in sys.argv
run = out.parent / f"{out.name}-run"


def student(path, labels, multi=False):
    model = AutoModelForSequenceClassification.from_pretrained(
        ENCODER,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        ignore_mismatched_sizes=True,
        problem_type="multi_label_classification" if multi else None,
    )
    model.save_pretrained(path / "student")
    AutoTokenizer.from_pretrained(ENCODER).save_pretrained(path / "student")
    export_onnx(path / "student", path / "onnx", int8=True)


student(run, ["negative", "positive"], multi)
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
if multi:  # the multi-label model has no PyTorch student (removed above), so its plain page is its only page
    print(build_site(run / "onnx", out, title="Browser test"))
    raise SystemExit

# a multi-label model with an explainer, at out/explainer-multi, and on the local Hub as test/multi
multi = out.parent / f"{out.name}-multi"
student(multi, ["sports", "politics", "money"], multi=True)
row = {"text": "the minister bought the football club", "gold": [0, 1, 2], "scores": [1.5, 2.0, 0.3]}
(multi / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
print(build_explainer(multi / "onnx", out / "explainer-multi", title="Browser test, multi-label"))
results["student"] = "a second student"
results["models"][1] |= {"macro_f1": 0.7}
(multi / "results.json").write_text(json.dumps(results), encoding="utf-8")
(multi / "meta.json").write_text(json.dumps({"task": "news topics"}), encoding="utf-8")

# a model without the PyTorch student, so without an explainer, on the local Hub as test/plain
plain = out.parent / f"{out.name}-plain"
student(plain, ["spam", "ham"])
shutil.rmtree(plain / "student")
row = {"text": "win a free phone now", "gold": 0}
(plain / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
(plain / "meta.json").write_text(json.dumps({"task": "spam or ham"}), encoding="utf-8")

# the local Hub, laid out as the Hub serves a repository: hub/<repo>/resolve/main/<file>
hub = out / "hub"
shutil.rmtree(hub, ignore_errors=True)
for repo, folder in (("test/multi", multi), ("test/plain", plain)):
    files = repo_files(folder / "onnx", folder / "browser")
    for name, f in files.items():
        (hub / repo / "resolve/main" / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, hub / repo / "resolve/main" / name)
(hub / "test/broken/resolve/main").mkdir(parents=True)
broken = json.loads((hub / "test/multi/resolve/main/tinydistill-page.json").read_text(encoding="utf-8"))
broken["simple"]["text"]["title"] = broken["explainer"]["text"]["title"] = "A broken model"
(hub / "test/broken/resolve/main/tinydistill-page.json").write_text(json.dumps(broken), encoding="utf-8")


def download(repo, filename):  # hf_hub_download from the local Hub
    f = hub / repo / "resolve/main" / filename
    if not f.exists():
        raise EntryNotFoundError(f"{repo} has no {filename}")
    return str(f)


huggingface_hub.hf_hub_download = download
# as `tinydistill web` does: the explainer is the main page and the plain page goes to simple/
models = "hf:test/multi,hf:test/plain,hf:test/broken"
main(["web", str(run / "onnx"), "--out", str(out), "--title", "Browser test", "--models", models])
main(["web", "hf:test/multi", "--out", str(out / "remote"), "--remote", "--models", "hf:test/plain"])
