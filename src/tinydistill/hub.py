"""Share an exported model on the Hugging Face Hub, with a generated model card."""

from __future__ import annotations

import json
from pathlib import Path


def model_card(onnx_dir: Path, repo: str, license: str | None = None) -> str:
    """A README.md for the exported model: what it is, its labels, the results table if the run has one,
    and how to use it."""
    run = onnx_dir.parent
    labels = json.loads((onnx_dir / "labels.json").read_text(encoding="utf-8"))
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    results = (run / "results.md").read_text() if (run / "results.md").exists() else ""
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
        "from huggingface_hub import snapshot_download",
        "from tinydistill import Predictor",
        "",
        f'clf = Predictor(snapshot_download("{repo}"))',
        'clf.predict(["some text"])',
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


def push(onnx_dir: Path, repo: str, private: bool = False, license: str | None = None) -> str:
    """Upload an exported model directory to the Hub (needs `huggingface-cli login` or HF_TOKEN)."""
    from huggingface_hub import HfApi

    if not (onnx_dir / "labels.json").exists():
        raise SystemExit(f"{onnx_dir} is not an exported model: run `tinydistill export` first")
    (onnx_dir / "README.md").write_text(model_card(onnx_dir, repo, license), encoding="utf-8")
    api = HfApi()
    url = api.create_repo(repo, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(onnx_dir), repo_id=repo, commit_message="Upload model from tinydistill")
    return str(url)
