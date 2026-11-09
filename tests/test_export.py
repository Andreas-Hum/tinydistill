import numpy as np
import onnx
import pytest
import torch
from conftest import TINY_ENCODER
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tinydistill.export import Predictor, export_onnx
from tinydistill.student import student_logits


def test_onnx_matches_pytorch(tmp_path):
    torch.manual_seed(0)
    labels = ["a", "b", "c"]
    model = AutoModelForSequenceClassification.from_pretrained(
        TINY_ENCODER, num_labels=3, id2label=dict(enumerate(labels)), ignore_mismatched_sizes=True
    ).eval()
    tok = AutoTokenizer.from_pretrained(TINY_ENCODER)
    model.save_pretrained(tmp_path / "student")
    tok.save_pretrained(tmp_path / "student")

    written = export_onnx(tmp_path / "student", tmp_path / "onnx", int8=True)
    assert [p.name for p in written] == ["model.onnx", "model.int8.onnx"]

    texts = ["short", "a somewhat longer input sentence to force padding", "third one here"]
    want = student_logits(model, tok, texts).numpy()
    fp32 = Predictor(tmp_path / "onnx", int8=False)
    np.testing.assert_allclose(fp32.logits(texts), want, atol=1e-4, rtol=1e-4)
    # batch size 1 (no padding) agrees too
    np.testing.assert_allclose(fp32.logits(texts, batch_size=1), want, atol=1e-4, rtol=1e-4)

    int8 = Predictor(tmp_path / "onnx")  # picks the int8 model when present
    assert int8.file.name == "model.int8.onnx"
    # one scale per output channel of each weight matrix, not one per matrix
    weights = onnx.load(int8.file).graph.initializer
    assert any(w.name.endswith("_scale") and list(w.dims) not in ([], [1]) for w in weights)
    assert int8.logits(texts).shape == (3, 3)
    assert fp32.predict(texts)[0] in labels
    np.testing.assert_allclose(fp32.predict_proba(texts).sum(-1), 1.0, atol=1e-5)


def test_static_student_trains_exports_and_predicts(tmp_path):
    from tinydistill.static import token_vectors
    from tinydistill.student import load_student, train_student

    texts = ["good movie", "bad film", "great", "awful"] * 4
    probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]] * 8)
    train_student(
        texts,
        probs,
        ["pos", "neg"],
        tmp_path / "s",
        f"static:{TINY_ENCODER}",
        epochs=5,
        lr=1e-2,
        batch_size=4,
    )
    model, tok = load_student(tmp_path / "s")
    assert type(model).__name__ == "StaticForSequenceClassification"
    v = token_vectors(TINY_ENCODER)
    assert v.shape == (len(tok), model.config.dim) and torch.isfinite(v).all()
    export_onnx(tmp_path / "s", tmp_path / "onnx", int8=True)
    p = Predictor(tmp_path / "onnx", int8=False)
    np.testing.assert_allclose(p.logits(texts[:4]), student_logits(model, tok, texts[:4]).numpy(), atol=1e-5)
    assert p.predict(texts[:4]) == ["pos", "neg", "pos", "neg"]


def test_explain_export_matches_the_model(tmp_path):
    import onnxruntime as ort

    from tinydistill.export import export_explain

    torch.manual_seed(0)
    model = AutoModelForSequenceClassification.from_pretrained(
        TINY_ENCODER, num_labels=3, id2label={0: "a", 1: "b", 2: "c"}, ignore_mismatched_sizes=True
    ).eval()
    tok = AutoTokenizer.from_pretrained(TINY_ENCODER)
    model.save_pretrained(tmp_path / "student")
    tok.save_pretrained(tmp_path / "student")
    path = export_explain(tmp_path / "student", tmp_path / "onnx")
    texts = ["a short one", "a much longer text than the first one"]
    enc = tok(texts, padding=True, return_tensors="np")
    feed = {k: enc[k].astype(np.int64) for k in ("input_ids", "attention_mask")}
    logits, layers, attention = ort.InferenceSession(str(path)).run(None, feed)
    np.testing.assert_allclose(logits, student_logits(model, tok, texts).numpy(), atol=1e-4)
    n_layers = model.config.num_hidden_layers
    assert layers.shape == (2, n_layers, 3) and np.allclose(layers[:, -1], logits)
    assert attention.shape == (2, n_layers, enc["input_ids"].shape[1])
    real = enc["attention_mask"].sum(1)
    np.testing.assert_allclose(attention[0, :, : real[0]].sum(-1), 1, atol=1e-4)  # a distribution over tokens
    assert np.allclose(attention[0, :, real[0] :], 0, atol=1e-6)  # padding gets none
    assert export_explain(tmp_path / "student", tmp_path / "q", int8=True).name == "explain.int8.onnx"


@pytest.mark.parametrize("arch", ["Bert", "DebertaV2", "Albert", "Electra", "MPNet", "Roberta"])
def test_explain_head_gives_the_models_logits(arch):
    """The head export_explain applies to each layer must give the model's own logits on the last layer, or
    the web page (which runs explain.onnx as the model) answers differently from Predictor."""
    import transformers

    from tinydistill.export import _classifier_head

    tok = AutoTokenizer.from_pretrained(TINY_ENCODER)
    config = getattr(transformers, f"{arch}Config")(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        num_labels=3,
        pad_token_id=tok.pad_token_id,
    )
    torch.manual_seed(0)
    model = getattr(transformers, f"{arch}ForSequenceClassification")(config).eval()
    head = _classifier_head(model)
    if head is None:
        return  # no layer view for this model; the page runs the exported model itself
    enc = tok(["a short one", "a much longer text than the first one"], padding=True, return_tensors="pt")
    enc = {k: enc[k] for k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        last = model.base_model(**enc).last_hidden_state
        torch.testing.assert_close(head(last), model(**enc).logits)


def test_int8_scores_do_not_depend_on_the_batch(tmp_path):
    torch.manual_seed(0)
    model = AutoModelForSequenceClassification.from_pretrained(
        TINY_ENCODER, num_labels=3, id2label={0: "a", 1: "b", 2: "c"}, ignore_mismatched_sizes=True
    )
    model.save_pretrained(tmp_path / "student")
    AutoTokenizer.from_pretrained(TINY_ENCODER).save_pretrained(tmp_path / "student")
    export_onnx(tmp_path / "student", tmp_path / "onnx", int8=True)
    p = Predictor(tmp_path / "onnx")
    assert p.int8
    text = "the same text every time"
    alone = p.logits([text])[0]
    for others in (["short"], ["a much longer text with many more words in it than the first one", "x"]):
        np.testing.assert_allclose(p.logits([text, *others])[0], alone, rtol=0, atol=1e-6)


def test_the_trained_input_length_travels_with_the_model(tmp_path):
    from tinydistill.student import load_student, train_student

    texts = ["word " * 50, "short"] * 4
    probs = torch.tensor([[0.7, 0.3], [0.3, 0.7]] * 4)
    train_student(texts, probs, ["a", "b"], tmp_path / "s", TINY_ENCODER, epochs=1, max_length=20)
    model, tok = load_student(tmp_path / "s")
    assert model.config.tinydistill_max_length == 20
    export_onnx(tmp_path / "s", tmp_path / "onnx")
    p = Predictor(tmp_path / "onnx")
    assert p.max_length == 20 and Predictor(tmp_path / "onnx", max_length=64).max_length == 64
    # the default cut is the trained one, so ONNX and PyTorch agree on long texts too
    np.testing.assert_allclose(p.logits(texts[:2]), student_logits(model, tok, texts[:2]).numpy(), atol=1e-5)
    assert not np.allclose(
        student_logits(model, tok, texts[:1]).numpy(),
        student_logits(model, tok, texts[:1], max_length=64).numpy(),
    )


def test_export_without_int8_removes_an_old_int8_model(tmp_path):
    model = AutoModelForSequenceClassification.from_pretrained(TINY_ENCODER, num_labels=2)
    model.save_pretrained(tmp_path / "student")
    AutoTokenizer.from_pretrained(TINY_ENCODER).save_pretrained(tmp_path / "student")
    export_onnx(tmp_path / "student", tmp_path / "onnx", int8=True)
    export_onnx(tmp_path / "student", tmp_path / "onnx")  # retrained and exported again, without --int8
    assert not (tmp_path / "onnx" / "model.int8.onnx").exists()  # Predictor would pick the old model
