import json
import re

import numpy as np
import onnxruntime as ort
import pytest
import torch
from conftest import TINY_ENCODER
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tinydistill.explainer import build_explainer
from tinydistill.export import EXPLORER_OUTPUTS, export_explorer, export_onnx
from tinydistill.student import student_logits


def _student(path, labels=("a", "b", "c"), **kw):
    torch.manual_seed(0)
    model = AutoModelForSequenceClassification.from_pretrained(
        TINY_ENCODER,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        ignore_mismatched_sizes=True,
        **kw,
    ).eval()
    tok = AutoTokenizer.from_pretrained(TINY_ENCODER)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    return model, tok


def test_explorer_export_matches_the_model(tmp_path):
    _, tok = _student(tmp_path / "student")
    path = export_explorer(tmp_path / "student", tmp_path / "onnx")
    texts = ["a short one", "a much longer text than the first one"]
    enc = tok(texts, padding=True, return_tensors="np")
    feed = {k: enc[k].astype(np.int64) for k in ("input_ids", "attention_mask")}
    session = ort.InferenceSession(str(path))
    assert [o.name for o in session.get_outputs()] == EXPLORER_OUTPUTS
    got = dict(zip(EXPLORER_OUTPUTS, session.run(None, feed), strict=True))

    model = AutoModelForSequenceClassification.from_pretrained(
        tmp_path / "student", attn_implementation="eager"
    ).eval()
    ids, mask = torch.tensor(feed["input_ids"]), torch.tensor(feed["attention_mask"])
    with torch.no_grad():
        out = model.bert(
            input_ids=ids, attention_mask=mask, output_attentions=True, output_hidden_states=True
        )
        hs = torch.stack(out.hidden_states, 1)
        close = lambda a, b: np.testing.assert_allclose(a, np.asarray(b), atol=1e-4)  # noqa: E731
        close(got["logits"], student_logits(model, tok, texts))
        close(got["layer_logits"][:, -1], got["logits"])
        close(got["attentions"], torch.stack(out.attentions, 1))
        close(got["hidden_states"], hs)
        close(got["token_embeddings"], model.bert.embeddings.word_embeddings(ids))
        # token + position (and segment) embeddings, through the embeddings' LayerNorm, give the first state
        summed = torch.tensor(got["token_embeddings"] + got["position_embeddings"])
        close(model.bert.embeddings.LayerNorm(summed), hs[:, 0])
        # the feed-forward block of each layer turns its attention state into the layer's output
        for k, layer in enumerate(model.bert.encoder.layer):
            a = torch.tensor(got["attention_states"][:, k])
            close(layer.output(layer.intermediate(a), a), hs[:, k + 1])
        close(got["head_input"], model.bert.pooler(hs[:, -1]))
        close(got["head_terms"].sum(-1) + got["head_bias"], got["logits"])
    n_layers, heads = model.config.num_hidden_layers, model.config.num_attention_heads
    assert got["attentions"].shape == (2, n_layers, heads, ids.shape[1], ids.shape[1])
    assert got["head_terms"].shape == (2, 3, model.config.hidden_size)
    assert export_explorer(tmp_path / "student", tmp_path / "q", int8=True).name == "explorer.int8.onnx"


@pytest.mark.parametrize("multi", [False, True])
def test_explainer_page(tmp_path, multi):
    run = tmp_path / "run"
    kw = {"problem_type": "multi_label_classification"} if multi else {}
    _student(run / "student", ("neg", "pos"), **kw)
    export_onnx(run / "student", run / "onnx", int8=True)
    rows = [{"text": 'say "<b>hi</b>"', "gold": [1] if multi else 0, "scores": [-1.0, -0.5]}]
    rows += [{"text": f"neg {i}", "gold": [0] if multi else 1} for i in range(3)]
    (run / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "results.json").write_text(
        json.dumps({"teacher": "T", "models": [], "training": {"ce_weight": 0.5, "n_gold": 20}})
    )
    page = build_explainer(run / "onnx", tmp_path / "site" / "explainer", title="Demo", note="Data <x>")
    (folder,) = [p for p in page.parent.iterdir() if p.is_dir()]
    assert (folder / "onnx" / "model_quantized.onnx").exists() and (folder / "tokenizer.json").exists()
    session = ort.InferenceSession(str(folder / "onnx" / "model_quantized.onnx"))
    assert [o.name for o in session.get_outputs()] == EXPLORER_OUTPUTS
    feed = {"input_ids": np.array([[2, 10, 11, 3]]), "attention_mask": np.ones((1, 4), dtype=np.int64)}
    got = dict(zip(EXPLORER_OUTPUTS, session.run(None, feed), strict=True))  # the int8 model runs
    np.testing.assert_allclose(got["head_terms"].sum(-1) + got["head_bias"], got["logits"], atol=0.05)
    text = page.read_text(encoding="utf-8")
    assert "<title>Demo: inside the model</title>" in text and "Data &lt;x&gt;" in text
    assert "<b>hi</b>" not in text and "<b>hi<\\/b>" in text  # "<\/" in the JSON data
    data = json.loads(text.split('id="data">')[1].split("</script>")[0])
    assert data["model"] == folder.name and data["dtype"] == "q8" and data["multi"] is multi
    assert data["config"]["num_attention_heads"] and data["max_length"] <= 128
    assert data["training"] == {"ce_weight": 0.5, "n_gold": 20}
    assert len(data["samples"][0]["teacher"]) == 2  # every label's soft label, for the loss in the page
    assert "katex@0.16." in text and not re.search(r"__[A-Z_]+__", text)  # every placeholder filled
    assert '<a href="simple/" data-base="simple/">Simple</a>' in text and 'id="tour-start"' in text


def test_explainer_needs_the_pytorch_student(tmp_path):
    _student(tmp_path / "student")
    export_onnx(tmp_path / "student", tmp_path / "onnx")
    (tmp_path / "student" / "config.json").unlink()
    with pytest.raises(ValueError, match="PyTorch student"):
        build_explainer(tmp_path / "onnx", tmp_path / "site")


def test_web_makes_the_explainer_the_main_page(tmp_path):
    from tinydistill.cli import main

    run = tmp_path / "run"
    _student(run / "student", ("neg", "pos"))
    export_onnx(run / "student", run / "onnx", int8=True)
    site = tmp_path / "site"
    (site / "explainer" / "model-0123456789").mkdir(parents=True)  # an explainer where it was before 0.4
    (site / "explainer" / "index.html").write_text("the old explainer")
    main(["web", str(run / "onnx"), "--out", str(site)])
    assert 'aria-current="page">Inside the model' in (site / "index.html").read_text(encoding="utf-8")
    assert 'aria-current="page">Simple' in (site / "simple" / "index.html").read_text(encoding="utf-8")
    old = site / "explainer" / "index.html"  # the old address forwards, with its ?text= and ?tour=1
    assert 'location.replace("../" + location.search' in old.read_text(encoding="utf-8")
    assert [p.name for p in (site / "explainer").iterdir()] == ["index.html"]
