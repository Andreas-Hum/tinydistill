import json

import numpy as np
import pytest
import torch
from conftest import TINY_ENCODER
from test_cli import TASK, write_jsonl
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tinydistill.cli import main
from tinydistill.export import Predictor
from tinydistill.student import student_logits, trim_vocab

TEXTS = [
    "Hvad er vejret i morgen?",
    "Sæt en alarm til klokken syv",
    "play some jazz music please",
    "wake me up",
]
OOV = ["日本語のテキスト Ωμέγα zzqx", "ÆØÅ æøå"]


def _tiny(family):
    """A random tiny classifier with a real tokenizer of each family: WordPiece (BERT), Unigram (the XLM-R
    tokenizer of the multilingual MiniLM), byte-fallback BPE (mmBERT's Gemma tokenizer) and byte-level BPE
    (ModernBERT's)."""
    from transformers import (
        ModernBertConfig,
        ModernBertForSequenceClassification,
        XLMRobertaConfig,
        XLMRobertaForSequenceClassification,
    )

    torch.manual_seed(0)
    if family == "wordpiece":
        model = AutoModelForSequenceClassification.from_pretrained(
            TINY_ENCODER, num_labels=3, ignore_mismatched_sizes=True
        )
        return model.eval(), AutoTokenizer.from_pretrained(TINY_ENCODER)
    small = dict(hidden_size=16, intermediate_size=32, num_attention_heads=2, num_labels=3)
    if family == "unigram":
        tok = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        config = XLMRobertaConfig(
            vocab_size=len(tok), num_hidden_layers=1, pad_token_id=tok.pad_token_id, **small
        )
        return XLMRobertaForSequenceClassification(config).eval(), tok
    tok = AutoTokenizer.from_pretrained(
        "jhu-clsp/mmBERT-small" if family == "bpe" else "answerdotai/ModernBERT-base"
    )
    ids = {k: getattr(tok, k) for k in ("pad_token_id", "cls_token_id", "sep_token_id")}
    config = ModernBertConfig(
        vocab_size=len(tok),
        num_hidden_layers=2,
        global_attn_every_n_layers=2,
        local_attention=16,
        **ids,
        **small,
    )
    return ModernBertForSequenceClassification(config).eval(), tok


@pytest.mark.parametrize("family", ["wordpiece", "unigram", "bpe", "bytelevel"])
def test_trimmed_student_agrees_on_kept_tokens(family, tmp_path):
    """Trimmed to the tokens of TEXTS, the model reads them as the same tokens (at new ids) and gives the same
    logits; the vocabulary shrinks; texts with trimmed tokens still run, through the unknown token or (BPE
    with byte fallback) bytes."""
    model, tok = _tiny(family)
    want = student_logits(model, tok, TEXTS)
    before = model.config.vocab_size
    new, rows = trim_vocab(model, tok, TEXTS, tmp_path, keep_top=20)
    assert rows == before and model.config.vocab_size == len(new) < min(1000, before // 5)
    assert model.get_input_embeddings().num_embeddings == len(new)
    old_ids, new_ids = tok(TEXTS)["input_ids"], new(TEXTS)["input_ids"]
    assert [tok.convert_ids_to_tokens(i) for i in old_ids] == [new.convert_ids_to_tokens(i) for i in new_ids]
    torch.testing.assert_close(student_logits(model, new, TEXTS), want, rtol=0, atol=1e-6)
    ids = new(OOV)["input_ids"]
    if family == "bpe":
        assert "<0xCE>" in new.convert_ids_to_tokens(ids[0])  # Greek letters, as bytes
    elif family == "bytelevel":  # no unknown token: every byte has a token of its own
        assert new.decode(ids[0], skip_special_tokens=True) == OOV[0]
    else:
        assert new.unk_token_id in ids[0] and new.unk_token_id in ids[1]
    assert student_logits(model, new, OOV).shape == (2, 3)
    assert all(i < len(new) for row in ids for i in row)


def test_train_trim_vocab_exports_and_records(tmp_path, capsys):
    """`train --trim-vocab`: the student trains trimmed to the training texts' tokens, the ONNX
    model, Predictor and the web page use the trimmed tokenizer, and meta.json records the setting."""
    from tinydistill.web import build_site

    run = tmp_path / "run"
    run.mkdir()
    (run / "task.yaml").write_text(TASK)
    pairs = [("good", 0, 2.0), ("bad", 1, -2.0)]
    train = [{"text": f"it was {w} {i}", "gold": g, "scores": [s, -s]} for i in range(4) for w, g, s in pairs]
    write_jsonl(run / "train.jsonl", train)
    write_jsonl(run / "test.jsonl", [{"text": "an awful film", "gold": 1, "scores": [-1.0, 1.0]}])
    common = ["--student", TINY_ENCODER, "--epochs", "1", "--device", "cpu"]
    main(["train", str(run), *common, "--trim-vocab", "--trim-keep", "10"])
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    config = json.loads((run / "student" / "config.json").read_text(encoding="utf-8"))
    assert meta["trim_vocab"] and meta["trim_keep"] == 10 and meta["vocab_size"] == config["vocab_size"] < 100
    assert "vocabulary trimmed to" in capsys.readouterr().err
    tok = AutoTokenizer.from_pretrained(run / "student")
    # only the training texts count: pieces that only the test text has ("awful film") are not kept
    assert {"##w", "##f", "##m"} - set(tok.get_vocab()) and "##d" in tok.get_vocab()

    main(["export", str(run), "--int8"])
    texts = ["it was good 1", "an awful film", "zebra quantum 日本"]
    model = AutoModelForSequenceClassification.from_pretrained(run / "student").eval()
    want = student_logits(model, tok, texts).numpy()
    np.testing.assert_allclose(Predictor(run / "onnx", int8=False).logits(texts), want, atol=1e-4)
    assert len(Predictor(run / "onnx").predict(texts)) == 3
    page = build_site(run / "onnx", tmp_path / "site")
    assert f"{config['vocab_size']:,} pieces" in page.read_text(encoding="utf-8")

    main(["train", str(run), *common])  # without the flag: the full vocabulary
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    assert meta["trim_vocab"] is False and meta["trim_keep"] is None and meta["vocab_size"] == 1124


def test_static_student_trims(tmp_path):
    """The static student's token vectors are trimmed as a transformer's embedding matrix is."""
    from tinydistill.static import new_static

    model, tok = new_static(TINY_ENCODER, ["a", "b", "c"], False, dim=8)
    want = student_logits(model, tok, TEXTS)
    new, rows = trim_vocab(model, tok, TEXTS, tmp_path, keep_top=0)
    assert model.embed.num_embeddings == model.config.vocab_size == len(new) < rows
    torch.testing.assert_close(student_logits(model, new, TEXTS), want, rtol=0, atol=1e-6)
