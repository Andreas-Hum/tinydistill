"""The explainer page of `tinydistill web`: a one-screen playground of the student as it runs in the browser
on the visitor's text, from the tokens through every layer and head to the probabilities, with the math of
each stage and what the student was trained on. The page itself is explainer.html next to this file;
the model is ``explorer.onnx`` (see export.export_explorer)."""

from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import tempfile
from pathlib import Path

from .web import TRANSFORMERS_JS, _samples, _summary, _teacher_probs

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.47/dist"


def build_explainer(
    onnx_dir: Path, out: Path, title: str | None = None, samples: int = 6, note: str | None = None
) -> Path:
    """Write ``out/index.html`` and ``out/model-<hash>/`` (config, tokenizer and ``explorer.onnx``). It needs
    the run's PyTorch student next to the export (``RUN/student``); raises ValueError, saying why, when the
    model has no explainer (no student folder, a static or ModernBERT-type student)."""
    from .export import export_explorer
    from .static import is_static

    run = onnx_dir.parent
    if not (onnx_dir / "config.json").exists():
        raise ValueError(f"{onnx_dir} has no config.json")
    if is_static(onnx_dir):
        raise ValueError("a static student has no layers to show")
    if not (run / "student" / "config.json").exists():
        raise ValueError(f"it needs the run's PyTorch student in {run / 'student'}")
    config = json.loads((onnx_dir / "config.json").read_text(encoding="utf-8"))
    int8 = (onnx_dir / "model.int8.onnx").exists()
    with tempfile.TemporaryDirectory() as tmp:
        onnx_file = export_explorer(run / "student", Path(tmp), int8)
        if onnx_file is None:
            raise ValueError(
                f"{config.get('model_type')} students have no explorer export (BERT-, RoBERTa- and "
                "MPNet-style students have)"
            )
        # named after the model's contents, as in build_site, so browsers never keep an older copy
        model_dir = out / f"model-{hashlib.sha256(onnx_file.read_bytes()).hexdigest()[:10]}"
        for old in out.glob("model-*"):
            if old.is_dir() and re.fullmatch(r"model-[0-9a-f]{10}", old.name):
                shutil.rmtree(old)
        (model_dir / "onnx").mkdir(parents=True)
        shutil.copy(onnx_file, model_dir / "onnx" / ("model_quantized.onnx" if int8 else "model.onnx"))
    for f in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ):
        if (onnx_dir / f).exists():
            shutil.copy(onnx_dir / f, model_dir / f)

    labels = json.loads((onnx_dir / "labels.json").read_text(encoding="utf-8"))
    multi = config.get("problem_type") == "multi_label_classification"
    read = lambda name: (  # noqa: E731
        json.loads((run / name).read_text(encoding="utf-8")) if (run / name).exists() else {}
    )
    meta, res = read("meta.json"), read("results.json")
    rows = []
    if (run / "test.jsonl").exists():
        with (run / "test.jsonl").open(encoding="utf-8") as fh:
            rows = _samples([json.loads(line) for line in fh if line.strip()], samples)
    keys = ["model_type", "num_hidden_layers", "num_attention_heads", "hidden_size", "intermediate_size"]
    keys += ["hidden_act", "layer_norm_eps", "vocab_size"]
    data = {
        "labels": labels,
        "multi": multi,
        "dtype": "q8" if int8 else "fp32",
        "model": model_dir.name,
        # every head's attention is tokens x tokens, so the page reads at most 128 tokens
        "max_length": min(config.get("tinydistill_max_length") or 128, 128),
        "config": {k: config.get(k) for k in keys},
        "teacher": meta.get("teacher") or res.get("teacher"),
        "training": {k: (res.get("training") or {}).get(k) for k in ("ce_weight", "n_gold")} if res else None,
        "samples": [
            {
                "text": r["text"],
                "gold": r.get("gold"),
                "teacher": _teacher_probs(r["scores"], multi, len(labels)) if r.get("scores") else None,
            }
            for r in rows
        ],
    }
    name = title or (meta.get("task") or run.name).replace("-", " ").replace("_", " ").title()
    page = (
        (Path(__file__).with_name("explainer.html").read_text(encoding="utf-8"))
        .replace("__TITLE__", html.escape(name))
        .replace("__SUMMARY__", html.escape(_summary(res or None, "int8" if int8 else "fp32")))
        .replace("__NOTE__", html.escape(note or ""))
        .replace("__TRANSFORMERS_JS__", TRANSFORMERS_JS)
        .replace("__KATEX__", KATEX)
        .replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    )
    (out / "index.html").write_text(page, encoding="utf-8")
    return out / "index.html"
