# `push` uploads what the browser pages need, and `web --models/--remote` builds pages that load models from
# the Hub. The Hub is mocked: HfApi copies the commit into a local folder, hf_hub_download reads from it.
import json
import shutil
from pathlib import Path

import huggingface_hub
import onnxruntime as ort
import pytest
import torch
from conftest import TINY_ENCODER
from huggingface_hub.errors import EntryNotFoundError
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tinydistill.cli import main
from tinydistill.export import EXPLORER_OUTPUTS, export_onnx


def _run(path: Path, labels=("neg", "pos"), multi=False, int8=True, student=True) -> Path:
    torch.manual_seed(0)
    kw = {"problem_type": "multi_label_classification"} if multi else {}
    model = AutoModelForSequenceClassification.from_pretrained(
        TINY_ENCODER,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        ignore_mismatched_sizes=True,
        **kw,
    )
    model.save_pretrained(path / "student")
    AutoTokenizer.from_pretrained(TINY_ENCODER).save_pretrained(path / "student")
    export_onnx(path / "student", path / "onnx", int8=int8)
    gold = (lambda i: [i % len(labels)]) if multi else (lambda i: i % len(labels))
    rows = [{"text": f"text {i}", "gold": gold(i), "scores": [-1.0] * len(labels)} for i in range(4)]
    (path / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    key = "macro_f1" if multi else "accuracy"
    results = {
        "teacher": "T",
        "student": "S",
        "data": {"n_test": 4},
        "models": [
            {"role": "teacher", "name": "teacher, 8-shot"},
            {"role": "student", "name": "int8" if int8 else "fp32", key: 0.5},
        ],
    }
    (path / "results.json").write_text(json.dumps(results))
    (path / "meta.json").write_text(json.dumps({"task": path.name, "teacher": "T", "student": "org/S"}))
    if not student:  # no PyTorch student: no explainer, and the plain export for the simple page
        shutil.rmtree(path / "student")
    return path / "onnx"


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A local stand-in for the Hub: hub/<user>/<name>/<file>, written by HfApi.create_commit."""
    root = tmp_path / "hub"

    class FakeApi:
        def create_repo(self, repo, private, exist_ok):
            return f"https://huggingface.co/{repo}"

        def create_commit(self, repo, operations, commit_message):
            with (root / "commits").open("a") as f:
                f.write(repo + "\n")
            for op in operations:
                (root / repo / op.path_in_repo).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(op.path_or_fileobj, root / repo / op.path_in_repo)

    def download(repo, filename):
        if not (root / repo / filename).exists():
            raise EntryNotFoundError(f"{repo} has no {filename}")
        return str(root / repo / filename)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    root.mkdir()
    (root / "commits").write_text("")  # one line per commit
    return root


def _files(folder: Path) -> set[str]:
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


def _outputs(path: Path) -> list[str]:
    return [o.name for o in ort.InferenceSession(str(path)).get_outputs()]


def _data(page: Path) -> dict:
    return json.loads(page.read_text(encoding="utf-8").split('id="data">')[1].split("</script>")[0])


def test_push_uploads_the_browser_files(tmp_path, hub):
    onnx = _run(tmp_path / "single")
    main(["push", str(onnx), "--repo", "me/single"])
    repo = hub / "me" / "single"
    assert (hub / "commits").read_text(encoding="utf-8") == "me/single\n"  # one commit
    export = {"model.onnx", "model.int8.onnx", "config.json", "labels.json", "tokenizer.json", "README.md"}
    assert export <= _files(repo)  # the export as it was, for Predictor("hf:...")
    assert _files(repo) - _files(onnx) == {
        "onnx/model_quantized.onnx",
        "onnx/explorer_quantized.onnx",
        "tinydistill-page.json",
    }
    assert _outputs(repo / "onnx/model_quantized.onnx") == ["logits", "layer_logits", "cls_attention"]
    assert _outputs(repo / "onnx/explorer_quantized.onnx") == EXPLORER_OUTPUTS
    page = json.loads((repo / "tinydistill-page.json").read_text(encoding="utf-8"))
    assert page["simple"]["labels"] == ["neg", "pos"] and page["simple"]["score"] == "test accuracy 0.500"
    assert page["explainer"]["file"] == "explorer" and page["explainer"]["dtype"] == "q8"
    assert "model" not in page["simple"] and "model" not in page["explainer"]  # no local folder

    # --title and --note: what the pages show for this model (the footer note swaps with the model)
    main(["push", str(onnx), "--repo", "me/single", "--title", "Reviews", "--note", "Data: CC BY 4.0."])
    page = json.loads((repo / "tinydistill-page.json").read_text(encoding="utf-8"))
    for p in ("simple", "explainer"):
        assert page[p]["text"]["title"] == "Reviews" and page[p]["text"]["note"] == "Data: CC BY 4.0."

    multi = _run(tmp_path / "multi", ("a", "b", "c"), multi=True, int8=False)
    main(["push", str(multi), "--repo", "me/multi"])
    assert _files(hub / "me" / "multi") - _files(multi) == {
        "onnx/model.onnx",  # fp32: no _quantized suffix
        "onnx/explorer.onnx",
        "tinydistill-page.json",
    }
    page = json.loads((hub / "me" / "multi" / "tinydistill-page.json").read_text(encoding="utf-8"))
    assert page["simple"]["multi"] and page["simple"]["score"] == "test macro-F1 0.500"
    assert page["explainer"]["dtype"] == "fp32" and len(page["explainer"]["labels"]) == 3

    plain = _run(tmp_path / "plain", student=False)
    main(["push", str(plain), "--repo", "me/plain"])
    assert _files(hub / "me" / "plain") - _files(plain) == {
        "onnx/model_quantized.onnx",
        "tinydistill-page.json",
    }
    assert _outputs(hub / "me" / "plain" / "onnx/model_quantized.onnx") == ["logits"]
    assert (
        json.loads((hub / "me" / "plain" / "tinydistill-page.json").read_text(encoding="utf-8"))["explainer"]
        is None
    )


def test_web_models_and_remote(tmp_path, hub, capsys):
    main(["push", str(_run(tmp_path / "other-task", ("x", "y", "z"))), "--repo", "me/other"])
    main(["push", str(_run(tmp_path / "plain", student=False)), "--repo", "me/plain"])
    onnx = _run(tmp_path / "mine")
    capsys.readouterr()
    site = tmp_path / "site"
    main(["web", str(onnx), "--out", str(site), "--models", "hf:me/other, hf:me/plain,hf:me/none"])
    assert "skipping hf:me/none: it has no tinydistill-page.json" in capsys.readouterr().err
    simple, inside = _data(site / "simple" / "index.html"), _data(site / "index.html")
    assert [(m["repo"], m["title"], m["explainer"]) for m in simple["models"]] == [
        (None, "Mine", True),
        ("me/other", "Other Task", True),
        ("me/plain", "Plain", False),
    ]
    assert simple["models"][1]["student"] == "S" and simple["models"][1]["score"] == "test accuracy 0.500"
    assert simple["models"][1]["size"].endswith(" MB")
    assert [m["repo"] for m in inside["models"]] == [
        None,
        "me/other",
        "me/plain",
    ]  # the page lists explainers
    assert 'id="picker"' in (site / "index.html").read_text(encoding="utf-8") and 'id="pick-list"' in (
        site / "simple/index.html"
    ).read_text(encoding="utf-8")

    # the page's data is the pushed page JSON, with where the model is
    page = json.loads((hub / "me" / "other" / "tinydistill-page.json").read_text(encoding="utf-8"))
    other = tmp_path / "other-site"
    main(["web", str(tmp_path / "other-task" / "onnx"), "--out", str(other)])
    simple, inside = _data(other / "simple" / "index.html"), _data(other / "index.html")
    assert {k: v for k, v in simple.items() if k not in ("model", "models")} == page["simple"]
    assert {k: v for k, v in inside.items() if k not in ("model", "models")} | {"file": "explorer"} == page[
        "explainer"
    ]

    # --remote: the same pages with no model files; the page loads its model from the Hub
    remote = tmp_path / "remote"
    (remote / "model-0123456789").mkdir(parents=True)  # an earlier local build's model is removed
    main(
        [
            "web",
            "hf:me/other",
            "--out",
            str(remote),
            "--remote",
            "--models",
            "hf:me/plain",
            "--title",
            "Remote",
        ]
    )
    assert _files(remote) == {"index.html", "simple/index.html"}
    simple, inside = _data(remote / "simple" / "index.html"), _data(remote / "index.html")
    assert simple["repo"] == inside["repo"] == "me/other" and "model" not in simple
    assert inside["file"] == "explorer" and simple["labels"] == ["x", "y", "z"]
    assert [m["repo"] for m in simple["models"]] == ["me/other", "me/plain"]
    assert '<h1 data-t="title">Remote</h1>' in (remote / "simple" / "index.html").read_text(encoding="utf-8")
    main(
        ["web", "hf:me/plain", "--out", str(tmp_path / "remote-plain"), "--remote"]
    )  # no explainer: one page
    assert _files(tmp_path / "remote-plain") == {"index.html"}
    with pytest.raises(SystemExit, match="no tinydistill-page.json"):
        main(["web", "hf:me/none", "--out", str(tmp_path / "x"), "--remote"])
    with pytest.raises(SystemExit, match="needs a model on the Hub"):
        main(["web", str(onnx), "--out", str(tmp_path / "x"), "--remote"])
