import json

import pytest
import torch
from conftest import TINY_ENCODER
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tinydistill.static import new_static
from tinydistill.student import init_head_from_names, load_student, student_logits, train_student

NAMES = ["sports and games", "money and banks", "the weather today"]
LABELS = ["sport", "money", "weather"]


def _bert():
    model = AutoModelForSequenceClassification.from_pretrained(
        TINY_ENCODER, num_labels=3, ignore_mismatched_sizes=True
    )
    return model, AutoTokenizer.from_pretrained(TINY_ENCODER)


def _roberta():  # the head is classifier.out_proj after a dense layer, as in MPNet
    from transformers import RobertaConfig, RobertaForSequenceClassification

    tok = AutoTokenizer.from_pretrained(TINY_ENCODER)
    config = RobertaConfig(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        pad_token_id=tok.pad_token_id,
        num_labels=3,
    )
    return RobertaForSequenceClassification(config), tok


def _modernbert():  # mean pooling, then a dense layer and norm, then the classifier
    from transformers import ModernBertConfig, ModernBertForSequenceClassification

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
        classifier_pooling="mean",
        pad_token_id=tok.pad_token_id,
        bos_token_id=tok.cls_token_id,
        eos_token_id=tok.sep_token_id,
        cls_token_id=tok.cls_token_id,
        sep_token_id=tok.sep_token_id,
        num_labels=3,
    )
    return ModernBertForSequenceClassification(config), tok


def _static():
    return new_static(TINY_ENCODER, LABELS, False)


@pytest.mark.parametrize("make", [_bert, _roberta, _modernbert, _static])
def test_head_starts_from_the_label_names(make):
    """Each row of the final linear layer is the normalised representation of a label's name that the layer
    receives, so before any training each name scores its own label highest."""
    torch.manual_seed(0)
    model, tok = make()
    init_head_from_names(model, tok, NAMES)
    head = [m for m in model.modules() if isinstance(m, torch.nn.Linear) and m.out_features == 3][-1]
    norms = head.weight.norm(dim=-1)
    torch.testing.assert_close(norms, norms[:1].expand(3))  # all rows the same length
    assert head.bias is None or not head.bias.any()
    assert student_logits(model.eval(), tok, NAMES).argmax(-1).tolist() == [0, 1, 2]


def test_train_student_saves_the_initialised_head(tmp_path):
    texts = ["good movie", "bad film", "great"] * 2
    train_student(
        texts,
        torch.full((6, 3), 1 / 3),
        LABELS,
        tmp_path / "s",
        TINY_ENCODER,
        epochs=1,
        lr=0.0,
        init_head=NAMES,
    )
    model, tok = load_student(tmp_path / "s")
    assert student_logits(model, tok, NAMES).argmax(-1).tolist() == [0, 1, 2]


def test_static_student_from_a_model2vec_folder(tmp_path):
    """A Model2Vec (potion) model: config.json with model_type model2vec, the token vectors as `embeddings` in
    model.safetensors, and a tokenizer. The static student starts from those vectors as they are; tokens
    without one start at zero."""
    from safetensors.torch import save_file

    tok = AutoTokenizer.from_pretrained(TINY_ENCODER)
    tok.save_pretrained(tmp_path / "potion")
    vectors = torch.randn(len(tok) - 1, 8)  # one token without a vector, as <s> in potion-multilingual
    save_file({"embeddings": vectors}, str(tmp_path / "potion" / "model.safetensors"))
    (tmp_path / "potion" / "config.json").write_text(
        json.dumps({"model_type": "model2vec", "hidden_dim": 8, "normalize": True}), encoding="utf-8"
    )
    model, _ = new_static(str(tmp_path / "potion"), LABELS, False)
    torch.testing.assert_close(model.embed.weight[:-1], vectors)
    assert model.config.vocab_size == len(tok) and not model.embed.weight[-1].any()
    texts = ["good movie", "bad film", "great"] * 2
    train_student(
        texts, torch.full((6, 3), 1 / 3), LABELS, tmp_path / "s", f"static:{tmp_path / 'potion'}", epochs=1
    )
    model, tok = load_student(tmp_path / "s")
    assert model.config.dim == 8 and student_logits(model, tok, texts).shape == (6, 3)
