"""The explainer page of `tinydistill web`: a one-screen playground of the student as it runs in the browser
on the visitor's text, from the tokens through every layer and head to the probabilities, with the math of
each stage and what the student was trained on. The page itself is explainer.html next to this file;
the model is ``explorer.onnx`` (see export.export_explorer)."""

from __future__ import annotations

import html
import json
import tempfile
from pathlib import Path

from .web import (
    PICKER_CSS,
    PICKER_HTML,
    PICKER_JS,
    TRANSFORMERS_JS,
    _outside,
    _read,
    _samples,
    _score,
    _summary,
    _teacher_probs,
    catalogue,
    copy_model,
)

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.47/dist"


def explorer_model(onnx_dir: Path, tmp: Path) -> Path:
    """The explainer's ONNX file (export.export_explorer, written to ``tmp``). It needs the run's PyTorch
    student next to the export (``RUN/student``); raises ValueError, saying why, when the model has no
    explainer (no student folder, a static or ModernBERT-type student)."""
    from .export import export_explorer
    from .static import is_static

    run = onnx_dir.parent
    if not (onnx_dir / "config.json").exists():
        raise ValueError(f"{onnx_dir} has no config.json")
    if is_static(onnx_dir):
        raise ValueError("a static student has no layers to show")
    if not (run / "student" / "config.json").exists():
        raise ValueError(f"it needs the run's PyTorch student in {run / 'student'}")
    onnx_file = export_explorer(run / "student", tmp, (onnx_dir / "model.int8.onnx").exists())
    if onnx_file is None:
        model_type = _read(onnx_dir / "config.json").get("model_type")
        raise ValueError(
            f"{model_type} students have no explorer export (BERT-, RoBERTa- and MPNet-style students have)"
        )
    return onnx_file


def explainer_data(onnx_dir: Path, size_mb: float, title: str | None = None, samples: int = 6) -> dict:
    """Everything the explainer page shows for a model, as the JSON it embeds, apart from where the model is
    (build_explainer adds the folder, ``model``; a page that loads it from the Hub adds ``repo``).
    ``size_mb``: the size of the explorer ONNX file. `push` uploads it in tinydistill-page.json."""
    run = onnx_dir.parent
    config, meta, res = _read(onnx_dir / "config.json"), _read(run / "meta.json"), _read(run / "results.json")
    labels = json.loads((onnx_dir / "labels.json").read_text(encoding="utf-8"))
    multi = config.get("problem_type") == "multi_label_classification"
    deployed = "int8" if (onnx_dir / "model.int8.onnx").exists() else "fp32"
    rows = []
    if (run / "test.jsonl").exists():
        with (run / "test.jsonl").open(encoding="utf-8") as fh:
            rows = _samples([json.loads(line) for line in fh if line.strip()], samples)
    keys = ["model_type", "num_hidden_layers", "num_attention_heads", "hidden_size", "intermediate_size"]
    keys += ["hidden_act", "layer_norm_eps", "vocab_size"]
    return {
        "labels": labels,
        "multi": multi,
        "dtype": "q8" if deployed == "int8" else "fp32",
        # every head's attention is tokens x tokens, so the page reads at most 128 tokens
        "max_length": min(config.get("tinydistill_max_length") or 128, 128),
        "outside": _outside(config, labels),
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
        "size_mb": round(size_mb, 2),
        "student": res.get("student") or meta.get("student"),
        "score": _score(res or None, deployed, multi),
        "text": {  # the page's text that depends on the model, by its data-t attribute
            "title": title or (meta.get("task") or run.name).replace("-", " ").replace("_", " ").title(),
            "summary": _summary(res or None, deployed),
        },
    }


def render_explainer(
    data: dict, out: Path, note: str | None = None, hub: dict[str, dict] | None = None
) -> Path:
    """Write ``out/index.html`` from a model's explainer data (explainer_data plus ``model`` or ``repo``).
    ``hub``: other models for the picker, repo -> tinydistill-page.json."""
    data = data | {"models": catalogue(data, True, hub or {}, "explainer")}
    page = (
        (Path(__file__).with_name("explainer.html").read_text(encoding="utf-8"))
        .replace("__TITLE__", html.escape(data["text"]["title"]))
        .replace("__SUMMARY__", html.escape(data["text"]["summary"]))
        .replace("__NOTE__", html.escape(note or ""))
        .replace("__PICKER_CSS__", PICKER_CSS)
        .replace("__PICKER__", PICKER_HTML)
        .replace("__PICKER_JS__", PICKER_JS)
        .replace("__TRANSFORMERS_JS__", TRANSFORMERS_JS)
        .replace("__KATEX__", KATEX)
        .replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page, encoding="utf-8")
    return out / "index.html"


def build_explainer(
    onnx_dir: Path,
    out: Path,
    title: str | None = None,
    samples: int = 6,
    note: str | None = None,
    hub: dict[str, dict] | None = None,
) -> Path:
    """Write ``out/index.html`` and ``out/model-<hash>/`` (config, tokenizer and ``explorer.onnx``). It needs
    the run's PyTorch student next to the export (``RUN/student``); raises ValueError, saying why, when the
    model has no explainer (no student folder, a static or ModernBERT-type student). ``hub``: other models for
    the model picker, Hub repo -> its tinydistill-page.json (see `tinydistill push`)."""
    with tempfile.TemporaryDirectory() as tmp:
        onnx_file = explorer_model(onnx_dir, Path(tmp))
        size = onnx_file.stat().st_size / 1e6
        folder = copy_model(onnx_dir, onnx_file, out)
    data = explainer_data(onnx_dir, size, title, samples) | {"model": folder}
    return render_explainer(data, out, note, hub)
