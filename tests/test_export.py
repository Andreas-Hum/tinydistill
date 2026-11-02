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
