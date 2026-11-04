"""A static-embedding student: one learned vector per token, averaged over the text, then a linear layer.
No attention, so it is far smaller and faster than a transformer. The token vectors start from a sentence
encoder (each token run through it on its own, then reduced with PCA), then train with the usual loss.

It is a regular `transformers` model, so training, ONNX export and `Predictor` work as for any student:
``--student static`` (from MiniLM) or ``--student static:<encoder>``. ``<encoder>`` can also be a Model2Vec
model (e.g. ``minishlab/potion-base-8M``): its token vectors are used as they are."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

PREFIX = "static"


class StaticConfig(PretrainedConfig):
    model_type = "tinydistill-static"

    def __init__(self, vocab_size: int = 1, dim: int = 256, **kw):
        super().__init__(**kw)
        self.vocab_size, self.dim = vocab_size, dim


class StaticForSequenceClassification(PreTrainedModel):
    config_class = StaticConfig

    def __init__(self, config: StaticConfig):
        super().__init__(config)
        self.embed = torch.nn.Embedding(config.vocab_size, config.dim)
        # transformers types num_labels as int | None; a config always has a label count here
        self.head = torch.nn.Linear(config.dim, config.num_labels)  # type: ignore[arg-type]
        self.post_init()

    def forward(self, input_ids, attention_mask=None, **_):
        m = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        m = m.unsqueeze(-1).to(self.embed.weight.dtype)
        mean = (self.embed(input_ids) * m).sum(1) / m.sum(1).clamp_min(1.0)
        return SequenceClassifierOutput(logits=self.head(mean))


AutoConfig.register(StaticConfig.model_type, StaticConfig)
AutoModelForSequenceClassification.register(StaticConfig, StaticForSequenceClassification)


def encoder_of(student: str) -> str | None:
    """The sentence encoder a static student starts from, or None for a transformer student."""
    if student == PREFIX:
        from .student import DEFAULT_STUDENT

        return DEFAULT_STUDENT
    return student.split(":", 1)[1] if student.startswith(PREFIX + ":") else None


@functools.lru_cache(maxsize=2)  # several seeds and variants of one run start from the same vectors
@torch.no_grad()
def token_vectors(encoder: str, dim: int = 256, batch: int = 1024) -> torch.Tensor:
    """Every vocabulary token through the encoder on its own (with its special tokens), mean-pooled, then PCA
    down to ``dim``."""
    tok = AutoTokenizer.from_pretrained(encoder)
    model = AutoModel.from_pretrained(encoder).eval()
    specials = tok("")["input_ids"]  # e.g. [CLS] [SEP]: the token goes after a leading CLS/BOS
    n = int(bool(specials) and specials[0] in (tok.cls_token_id, tok.bos_token_id))
    rows = []
    for b in range(0, len(tok), batch):
        x = torch.tensor([specials[:n] + [i] + specials[n:] for i in range(b, min(b + batch, len(tok)))])
        rows.append(model(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state.mean(1))
    v = torch.cat(rows)
    v = v - v.mean(0)
    _, _, vt = torch.linalg.svd(v, full_matrices=False)  # PCA: the top principal directions
    return v @ vt[:dim].T


def model2vec_vectors(encoder: str) -> torch.Tensor | None:
    """The token vectors of a Model2Vec model (a Hub id or folder with ``model_type: model2vec`` in its
    config.json and an ``embeddings`` matrix in model.safetensors), or None if ``encoder`` is not one."""
    from safetensors.torch import load_file
    from transformers.utils import cached_file

    config = cached_file(encoder, "config.json", _raise_exceptions_for_missing_entries=False)
    if not config or json.loads(Path(config).read_text(encoding="utf-8")).get("model_type") != "model2vec":
        return None
    return load_file(str(cached_file(encoder, "model.safetensors")))["embeddings"].float()


def new_static(encoder: str, labels: list[str], multi_label: bool, dim: int = 256):
    """A static student initialised from ``encoder`` (a Model2Vec model keeps its own vectors and size), and
    its tokenizer."""
    tok = AutoTokenizer.from_pretrained(encoder)
    vectors = model2vec_vectors(encoder)
    if vectors is None:
        vectors = token_vectors(encoder, dim)
    if len(tok) > len(vectors):  # tokens a Model2Vec model has no vector for (e.g. <s>, </s>) start at zero
        vectors = torch.cat([vectors, vectors.new_zeros(len(tok) - len(vectors), vectors.shape[1])])
    config = StaticConfig(
        vocab_size=vectors.shape[0],
        dim=vectors.shape[1],
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={name: i for i, name in enumerate(labels)},
        problem_type="multi_label_classification" if multi_label else "single_label_classification",
        pad_token_id=tok.pad_token_id,
    )
    model = StaticForSequenceClassification(config)
    model.embed.weight.data.copy_(vectors)
    return model, tok


def is_static(model_dir: Path) -> bool:
    f = Path(model_dir) / "config.json"
    return (
        f.exists() and json.loads(f.read_text(encoding="utf-8")).get("model_type") == StaticConfig.model_type
    )
