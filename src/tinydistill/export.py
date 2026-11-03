"""Export the student to ONNX (optionally int8 dynamic quantisation) and run it with onnxruntime."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from .student import load_student


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
        self, path: str | Path, int8: bool | None = None, threads: int | None = None, max_length: int = 128
    ):
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
        self.max_length = max_length
        self.file = file

    def logits(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
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

    def predict_proba(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Label probabilities per text: a softmax, or for a multi-label model each label's sigmoid."""
        z = self.logits(texts, batch_size)
        if self.multi_label:
            return 1 / (1 + np.exp(-z))
        z = np.exp(z - z.max(-1, keepdims=True))
        return z / z.sum(-1, keepdims=True)

    def predict(self, texts: list[str], batch_size: int = 32) -> list:
        """The top label per text, or for a multi-label model the list of labels with probability >= 0.5."""
        z = self.logits(texts, batch_size)
        if self.multi_label:
            return [[self.labels[i] for i in np.flatnonzero(row > 0)] for row in z]
        return [self.labels[i] for i in z.argmax(-1)]

    def describe(self, texts: list[str]) -> list[dict]:
        """Per text, as JSON-ready dicts: the label and its probability, or (multi-label) the labels and
        every label's probability."""
        out = []
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
            answered = self.teacher.score_many([texts[i] for i in unsure])
            if self.log:
                with self.log.open("a", encoding="utf-8") as f:
                    for i, scores in zip(unsure, answered, strict=True):
                        f.write(json.dumps({"text": texts[i], "scores": scores}, ensure_ascii=False) + "\n")
            for i, scores in zip(unsure, answered, strict=True):
                s = np.asarray(scores, dtype=np.float64)
                p = np.exp(s - s.max())
                p /= p.sum()
                out[i] = {
                    "label": self.student.labels[int(p.argmax())],
                    "confidence": float(p.max()),
                    "by": "teacher",
                }
        return out
