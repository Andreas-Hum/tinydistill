"""Share an exported model on the Hugging Face Hub, with a generated model card."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def model_card(onnx_dir: Path, repo: str, license: str | None = None) -> str:
    """A README.md for the exported model: what it is, its labels, the results table if the run has one,
    and how to use it."""
    run = onnx_dir.parent
    labels = json.loads((onnx_dir / "labels.json").read_text(encoding="utf-8"))
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8")) if (run / "meta.json").exists() else {}
    results = (run / "results.md").read_text(encoding="utf-8") if (run / "results.md").exists() else ""
    front = [
        "---",
        "library_name: onnx",
        "pipeline_tag: text-classification",
        "tags:",
        "- text-classification",
        "- onnx",
        "- knowledge-distillation",
        "- tinydistill",
    ]
    if license:
        front.append(f"license: {license}")
    student_id = str(meta.get("student", ""))
    if "/" in student_id and not student_id.startswith("static"):  # links the card to the encoder's page
        front.append(f"base_model: {student_id}")
    data = [str(meta.get(k) or "") for k in ("data", "test")]
    datasets = sorted({d.split(":")[1] for d in data if d.startswith("hf:") and len(d.split(":")) > 1})
    if datasets:  # links the card to the datasets' pages
        front += ["datasets:", *(f"- {d}" for d in datasets)]
    front.append("---")
    teacher, student = meta.get("teacher", "an LLM"), meta.get("student", "a small encoder")
    lines = [
        *front,
        "",
        f"# {repo.split('/')[-1]}",
        "",
        f"A small text classifier distilled from `{teacher}` into `{student}` with "
        "[tinydistill](https://github.com/Andreas-Hum/tinydistill), exported to ONNX for fast CPU inference.",
        "",
        f"Labels ({len(labels)}): " + ", ".join(f"`{x}`" for x in labels),
        "",
        "## Usage",
        "",
        "```python",
        "from tinydistill import Predictor  # pip install tinydistill",
        "",
        f'clf = Predictor("hf:{repo}")',
        'clf.predict(["some text"])',
        "```",
        "",
        "Or from the command line:",
        "",
        "```sh",
        f'tinydistill predict hf:{repo} "some text"',
        f"tinydistill serve hf:{repo}",
        "```",
        "",
    ]
    if results:
        lines += ["## Results", "", results.strip(), ""]
    lines += [
        "## License",
        "",
        "Check the licences of the student model, the teacher and the training data before using this model."
        + (f" The uploader chose `{license}`." if license else ""),
        "",
    ]
    return "\n".join(lines)


PAGE_FILE = "tinydistill-page.json"


def repo_files(onnx_dir: Path, tmp: Path) -> dict[str, Path]:
    """Every file `push` uploads, by its path in the repository: the export as it is (for ``Predictor``), and
    for the browser pages of `tinydistill web --models/--remote` the ONNX models in the transformers.js layout
    (``onnx/model_quantized.onnx`` for int8, else ``onnx/model.onnx``; the explainer's as
    ``onnx/explorer_quantized.onnx`` or ``onnx/explorer.onnx`` when the run's PyTorch student has one) and
    PAGE_FILE, the data each page embeds: {"simple": ..., "explainer": ... or null}. Static students have no
    page, so only the export. Files it writes go to ``tmp``."""
    from .explainer import explainer_data, explorer_model
    from .static import is_static
    from .web import site_data, site_model

    files = {p.relative_to(onnx_dir).as_posix(): p for p in sorted(onnx_dir.rglob("*")) if p.is_file()}
    if is_static(onnx_dir) or not (onnx_dir / "config.json").exists():
        return files
    suffix = "_quantized" if (onnx_dir / "model.int8.onnx").exists() else ""
    files[f"onnx/model{suffix}.onnx"] = model = site_model(onnx_dir, tmp / "simple")
    page = {"simple": site_data(onnx_dir, model.stat().st_size / 1e6), "explainer": None}
    try:
        files[f"onnx/explorer{suffix}.onnx"] = explorer_model(onnx_dir, tmp / "explainer")
        size = files[f"onnx/explorer{suffix}.onnx"].stat().st_size / 1e6
        page["explainer"] = explainer_data(onnx_dir, size) | {"file": "explorer"}  # model_file_name
    except ValueError as e:
        print(f"no explainer page: {e}", file=sys.stderr)
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / PAGE_FILE).write_text(json.dumps(page), encoding="utf-8")
    files[PAGE_FILE] = tmp / PAGE_FILE
    return files


def page_json(repo: str) -> dict | None:
    """A Hub model's PAGE_FILE (what the browser pages show for it), or None when it has none."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return json.loads(Path(hf_hub_download(repo, PAGE_FILE)).read_text(encoding="utf-8"))
    except EntryNotFoundError:
        return None


def push(onnx_dir: Path, repo: str, private: bool = False, license: str | None = None) -> str:
    """Upload an exported model directory to the Hub (needs `huggingface-cli login` or HF_TOKEN), with the
    files the browser pages need (repo_files), in one commit."""
    from huggingface_hub import CommitOperationAdd, HfApi

    if not (onnx_dir / "labels.json").exists():
        raise SystemExit(f"{onnx_dir} is not an exported model: run `tinydistill export` first")
    (onnx_dir / "README.md").write_text(model_card(onnx_dir, repo, license), encoding="utf-8")
    api = HfApi()
    url = api.create_repo(repo, private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        ops = [CommitOperationAdd(k, str(v)) for k, v in repo_files(onnx_dir, Path(tmp)).items()]
        api.create_commit(repo, ops, commit_message="Upload model from tinydistill")
    return str(url)
