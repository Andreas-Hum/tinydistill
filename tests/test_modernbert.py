import numpy as np
import torch

from tinydistill.export import Predictor, export_explain, export_explorer, export_onnx
from tinydistill.student import load_student, student_logits, train_student


def test_modernbert_student_trains_and_exports(tmp_path):
    """A ModernBERT student (built tiny from its config: there is no tiny test model) trains, exports to
    ONNX and int8, and the ONNX model gives the same logits as PyTorch. Its classifier reads a pooled sequence
    through its own head, so the browser page's layer view is not available (export_explain returns None)."""
    from transformers import AutoTokenizer, ModernBertConfig, ModernBertForSequenceClassification

    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    config = ModernBertConfig(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=512,
        global_attn_every_n_layers=2,
        local_attention=16,
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.cls_token_id,
        eos_token_id=tok.sep_token_id,
        cls_token_id=tok.cls_token_id,
        sep_token_id=tok.sep_token_id,
    )
    torch.manual_seed(0)
    ModernBertForSequenceClassification(config).save_pretrained(tmp_path / "base")
    tok.save_pretrained(tmp_path / "base")
    texts = ["good movie", "bad film", "a rather long text with many words in it " * 3, "awful"] * 4
    probs = torch.tensor([[0.9, 0.1], [0.1, 0.9]] * 8)
    train_student(
        texts, probs, ["pos", "neg"], tmp_path / "s", str(tmp_path / "base"), epochs=1, batch_size=4
    )
    model, tok = load_student(tmp_path / "s")
    want = student_logits(model, tok, texts[:4]).numpy()
    export_onnx(tmp_path / "s", tmp_path / "onnx", int8=True)
    np.testing.assert_allclose(Predictor(tmp_path / "onnx", int8=False).logits(texts[:4]), want, atol=1e-4)
    assert Predictor(tmp_path / "onnx").predict(texts[:4])  # the int8 model runs
    assert export_explain(tmp_path / "s", tmp_path / "explain") is None
    assert export_explorer(tmp_path / "s", tmp_path / "explorer") is None  # nor the explainer page
