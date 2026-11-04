"""Export the student to ONNX (optionally int8 dynamic quantisation) and run it with onnxruntime."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from .student import by_windows, load_student, windows


class _Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export_onnx(student_dir: Path, out_dir: Path, int8: bool = False, opset: int = 17) -> list[Path]:
    model, tok = load_student(student_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = tok(["an example input", "another one"], padding=True, return_tensors="pt")
    path = out_dir / "model.onnx"
    torch.onnx.export(
        _Logits(model),
        (enc["input_ids"], enc["attention_mask"]),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
        dynamo=False,
    )
    tok.save_pretrained(out_dir)
    model.config.save_pretrained(out_dir)  # config.json: labels for tools such as transformers.js
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    written = [path]
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q = out_dir / "model.int8.onnx"
        quantize_dynamic(str(path), str(q), weight_type=QuantType.QInt8)
        written.append(q)
    return written


def _classifier_head(model):
    """The student's classification head as a function of one layer's hidden states (batch, tokens, dim), or
    None when the model has no such head (the static student)."""
    head, base = getattr(model, "classifier", None), getattr(model, "base_model", None)
    if head is None or base is None or not hasattr(base, "encoder"):
        return None
    if isinstance(head, torch.nn.Linear):  # BERT style: the pooler reads the first token
        pooler = getattr(base, "pooler", None)
        return (lambda h: head(pooler(h))) if pooler is not None else (lambda h: head(h[:, 0]))
    return head  # RoBERTa/MPNet style heads take the whole sequence and read its first token themselves


class _Explain(torch.nn.Module):
    """Logits, plus what the classifier would answer after each layer, and how much the first token (which
    the classifier reads) attends to every token in each layer, averaged over heads."""

    def __init__(self, model, head):
        super().__init__()
        self.model, self.head = model, head

    def forward(self, input_ids, attention_mask):
        out = self.model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )
        hs = torch.stack(out.hidden_states[1:], 1)  # (batch, layers, tokens, dim)
        b, n, t, d = hs.shape
        # the head runs once on all layers stacked as a batch (one set of weights in the graph)
        layers = self.head(hs.reshape(b * n, t, d)).reshape(b, n, -1)  # (batch, layers, labels)
        attention = torch.stack([a[:, :, 0, :].mean(1) for a in out.attentions], 1)  # (batch, layers, tokens)
        return layers[:, -1], layers, attention


def export_explain(student_dir: Path, out_dir: Path, int8: bool = False, opset: int = 17) -> Path | None:
    """Export ``explain.onnx`` (and ``explain.int8.onnx``) with outputs ``logits``, ``layer_logits`` and
    ``cls_attention``, for the teaching view of `tinydistill web`. None if the model has no layers to show."""
    from transformers import AutoModelForSequenceClassification

    from .static import is_static

    if is_static(student_dir):
        return None
    model = AutoModelForSequenceClassification.from_pretrained(
        student_dir, attn_implementation="eager"
    ).eval()
    head = _classifier_head(model)
    if head is None:
        return None
    _, tok = load_student(student_dir)
    enc = tok(["an example input", "another one"], padding=True, return_tensors="pt")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "explain.onnx"
    torch.onnx.export(
        _Explain(model, head),
        (enc["input_ids"], enc["attention_mask"]),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits", "layer_logits", "cls_attention"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
            "layer_logits": {0: "batch"},
            "cls_attention": {0: "batch", 2: "seq"},
        },
        opset_version=opset,
        dynamo=False,
    )
    if not int8:
        return path
    from onnxruntime.quantization import QuantType, quantize_dynamic

    q = out_dir / "explain.int8.onnx"
    quantize_dynamic(str(path), str(q), weight_type=QuantType.QInt8)
    return q


EXPLORER_OUTPUTS = [
    "logits",  # (batch, labels)
    "layer_logits",  # (batch, layers, labels): what the classifier would answer after each layer
    "attentions",  # (batch, layers, heads, tokens, tokens): every head's attention weights
    "hidden_states",  # (batch, layers + 1, tokens, dim): after the embeddings and after each layer
    "attention_states",  # (batch, layers, tokens, dim): after each layer's attention, residual and LayerNorm
    "token_embeddings",  # (batch, tokens, dim): the token embedding of each token
    "position_embeddings",  # (batch, tokens, dim): what the embeddings add to it before their LayerNorm
    "head_input",  # (batch, dim): the vector h the classifier's last linear layer reads
    "head_terms",  # (batch, labels, dim): W * h, which summed over dim plus the bias gives the logits
    "head_bias",  # (batch, labels): that layer's bias b
]


class _Explore(torch.nn.Module):
    """Everything the explainer page draws for one text (see EXPLORER_OUTPUTS). Forward hooks read what the
    model's own outputs do not include: the input of the embeddings' LayerNorm, each layer's attention block
    and the input of the classifier's last linear layer."""

    def __init__(self, model, head, last):
        super().__init__()
        self.model, self.head, self.last = model, head, last

    def forward(self, input_ids, attention_mask):
        base, seen = self.model.base_model, {}
        hooks = [
            base.embeddings.LayerNorm.register_forward_hook(lambda m, i, o: seen.update(summed=i[0])),
            self.last.register_forward_pre_hook(lambda m, i: seen.update(head_input=i[0])),
        ] + [
            layer.attention.register_forward_hook(
                lambda m, i, o, k=k: seen.update({k: o[0] if isinstance(o, tuple) else o})
            )
            for k, layer in enumerate(base.encoder.layer)
        ]
        try:
            out = base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
            )
            hs = torch.stack(out.hidden_states, 1)  # (batch, layers + 1, tokens, dim)
            b, n, t, d = hs.shape
            layers = self.head(hs[:, 1:].reshape(b * (n - 1), t, d)).reshape(b, n - 1, -1)
        finally:
            for hook in hooks:
                hook.remove()
        h = seen["head_input"].reshape(b, n - 1, -1)[:, -1]  # the head ran on all layers; the last one's
        token = base.embeddings.word_embeddings(input_ids)
        return (
            layers[:, -1],
            layers,
            torch.stack(out.attentions, 1),
            hs,
            torch.stack([seen[k] for k in range(n - 1)], 1),
            token,
            seen["summed"] - token,
            h,
            h[:, None, :] * self.last.weight,
            self.last.bias.expand(b, -1),
        )


def export_explorer(student_dir: Path, out_dir: Path, int8: bool = False, opset: int = 17) -> Path | None:
    """Export ``explorer.onnx`` (and ``explorer.int8.onnx``) with the outputs in EXPLORER_OUTPUTS, for the
    explainer page of `tinydistill web`. None if the model has no such view (static and ModernBERT-type
    students): it needs a BERT-, RoBERTa- or MPNet-style encoder and head."""
    from transformers import AutoModelForSequenceClassification

    from .static import is_static

    if is_static(student_dir):
        return None
    model = AutoModelForSequenceClassification.from_pretrained(
        student_dir, attn_implementation="eager"
    ).eval()
    head = _classifier_head(model)
    if head is None:
        return None
    last = model.classifier
    last = last if isinstance(last, torch.nn.Linear) else getattr(last, "out_proj", None)
    emb = getattr(model.base_model, "embeddings", None)
    if (
        not isinstance(last, torch.nn.Linear)
        or not hasattr(model.base_model.encoder, "layer")
        or getattr(emb, "LayerNorm", None) is None
        or emb.word_embeddings.embedding_dim != model.config.hidden_size  # ELECTRA's projected embeddings
    ):
        return None
    _, tok = load_student(student_dir)
    enc = tok(["an example input", "another one"], padding=True, return_tensors="pt")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "explorer.onnx"
    seq = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}}
    axes = {"attentions": {3: "seq", 4: "seq"}, "hidden_states": {2: "seq"}, "attention_states": {2: "seq"}}
    axes |= {"token_embeddings": {1: "seq"}, "position_embeddings": {1: "seq"}}
    torch.onnx.export(
        _Explore(model, head, last),
        (enc["input_ids"], enc["attention_mask"]),
        str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=EXPLORER_OUTPUTS,
        dynamic_axes=seq | {k: {0: "batch"} | axes.get(k, {}) for k in EXPLORER_OUTPUTS},
        opset_version=opset,
        dynamo=False,
    )
    if not int8:
        return path
    from onnxruntime.quantization import QuantType, quantize_dynamic

    q = out_dir / "explorer.int8.onnx"
    quantize_dynamic(str(path), str(q), weight_type=QuantType.QInt8)
    return q


def model_path(path: str | Path) -> Path:
    """An exported model's folder: a local path, or ``hf:user/name`` for a model on the Hugging Face Hub (as
    uploaded by `tinydistill push`), downloaded once and then read from the local Hub cache."""
    if str(path).startswith("hf:"):
        from huggingface_hub import snapshot_download

        repo = str(path).removeprefix("hf:")
        return Path(snapshot_download(repo, allow_patterns=["*.onnx", "*.json", "*.txt", "*.model"]))
    if not Path(path).is_dir():
        raise FileNotFoundError(
            f"{path} is not a folder; for a model on the Hugging Face Hub write hf:user/name"
        )
    return Path(path)


class Predictor:
    """Run an exported model: ``Predictor("runs/x/onnx").predict(["text", ...])``, or one on the Hugging Face
    Hub: ``Predictor("hf:user/name")``."""

    def __init__(
        self,
        path: str | Path,
        int8: bool | None = None,
        threads: int | None = None,
        max_length: int | None = None,
    ):
        """``max_length``: tokens per input; by default the length the student was trained with (128 for
        models exported before it was recorded)."""
        import onnxruntime as ort
        from transformers import AutoTokenizer

        path = model_path(path)
        if int8 is None:
            int8 = (path / "model.int8.onnx").exists()
        file = path / ("model.int8.onnx" if int8 else "model.onnx")
        opts = ort.SessionOptions()
        threads = threads or int(os.environ.get("OMP_NUM_THREADS", "0"))
        if threads:
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(file), opts, providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(path)
        self.labels: list[str] = json.loads((path / "labels.json").read_text(encoding="utf-8"))
        config = (
            json.loads((path / "config.json").read_text(encoding="utf-8"))
            if (path / "config.json").exists()
            else {}
        )
        self.multi_label = config.get("problem_type") == "multi_label_classification"
        self.max_length: int = max_length or config.get("tinydistill_max_length") or 128
        # a model trained on windows of long documents reads every input in the same windows
        self.chunk_tokens: int | None = config.get("tinydistill_chunk_tokens")
        self.chunk_stride: int | None = config.get("tinydistill_chunk_stride")
        self.file = file
        self.int8 = int8

    def logits(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Label scores per text. The int8 model runs one text at a time by default: its dynamic quantisation
        scales activations over the whole batch, so in a batch a text's scores would depend on the other texts
        (on a CPU this costs little throughput). The fp32 model runs 32 texts at a time. A model trained on
        windows of long documents (``chunk_tokens``) scores each text's windows and returns, per text, the log
        of the mean of their probabilities (multi-label: of each label's sigmoid, as log-odds), so
        ``predict_proba`` gives the mean of the windows' probabilities."""
        batch_size = batch_size or (1 if self.int8 else 32)
        if self.chunk_tokens and texts:
            run = lambda w: self._logits(w, batch_size).tolist()  # noqa: E731
            return np.asarray(by_windows(run, texts, self.cut, self.multi_label), dtype=np.float32)
        return self._logits(texts, batch_size)

    def cut(self, text: str) -> list[str]:
        """The windows the model reads ``text`` in: the text itself unless it was trained on windows."""
        if not self.chunk_tokens:
            return [text]
        return windows(self.tok, text, self.chunk_tokens, self.chunk_stride)

    def _logits(self, texts: list[str], batch_size: int) -> np.ndarray:
        out = []
        for b in range(0, len(texts), batch_size):
            enc = self.tok(
                texts[b : b + batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            feed = {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            }
            out.append(self.session.run(["logits"], feed)[0])
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)), dtype=np.float32)

    def predict_proba(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Label probabilities per text: a softmax, or for a multi-label model each label's sigmoid."""
        z = self.logits(texts, batch_size)
        if self.multi_label:
            return 1 / (1 + np.exp(-z))
        z = np.exp(z - z.max(-1, keepdims=True))
        return z / z.sum(-1, keepdims=True)

    def predict(self, texts: list[str], batch_size: int | None = None) -> list:
        """The top label per text, or for a multi-label model the list of labels with probability >= 0.5."""
        z = self.logits(texts, batch_size)
        if self.multi_label:
            return [[self.labels[i] for i in np.flatnonzero(row > 0)] for row in z]
        return [self.labels[i] for i in z.argmax(-1)]

    def describe(self, texts: list[str]) -> list[dict]:
        """Per text, as JSON-ready dicts: the label and its probability, or (multi-label) the labels and
        every label's probability."""
        out: list[dict] = []
        for p in self.predict_proba(texts):
            if self.multi_label:
                out.append(
                    {
                        "labels": [self.labels[i] for i in np.flatnonzero(p >= 0.5)],
                        "scores": {lab: round(float(x), 4) for lab, x in zip(self.labels, p, strict=True)},
                    }
                )
            else:
                i = int(p.argmax())
                out.append({"label": self.labels[i], "confidence": round(float(p[i]), 4)})
        return out


class Cascade:
    """The student answers; texts where its top probability is below ``threshold`` go to the teacher.

    ``teacher`` is a LabelScorer (see ``load_teacher``) for the same task and label order."""

    def __init__(self, student: Predictor, teacher, threshold: float, log: str | Path | None = None):
        if student.multi_label:
            raise ValueError("the cascade is for single-label models")
        self.student, self.teacher, self.threshold = student, teacher, threshold
        self.log = Path(log) if log else None  # texts the teacher answered, for `tinydistill refresh`

    def predict(self, texts: list[str]) -> list[dict]:
        probs = self.student.predict_proba(texts)
        out = [
            {"label": self.student.labels[int(p.argmax())], "confidence": float(p.max()), "by": "student"}
            for p in probs
        ]
        unsure = [i for i, p in enumerate(probs) if p.max() < self.threshold]
        if unsure:
            answered = by_windows(self._ask, [texts[i] for i in unsure], self.student.cut)
            for i, scores in zip(unsure, answered, strict=True):
                s: np.ndarray = np.asarray(scores, dtype=np.float64)
                p = np.exp(s - s.max())
                p /= p.sum()
                out[i] = {
                    "label": self.student.labels[int(p.argmax())],
                    "confidence": float(p.max()),
                    "by": "teacher",
                }
        return out

    def _ask(self, texts: list[str]) -> list[list[float]]:
        """The teacher's scores (of a long document's windows, as the student reads it), logged if asked."""
        answered = self.teacher.score_many(texts)
        if self.log:
            with self.log.open("a", encoding="utf-8") as f:
                for text, scores in zip(texts, answered, strict=True):
                    f.write(json.dumps({"text": text, "scores": scores}, ensure_ascii=False) + "\n")
        return answered
