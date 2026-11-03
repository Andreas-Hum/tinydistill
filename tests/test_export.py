import numpy as np
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
