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
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    written = [path]
    if int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q = out_dir / "model.int8.onnx"
        quantize_dynamic(str(path), str(q), weight_type=QuantType.QInt8)
        written.append(q)
    return written


class Predictor:
    """Run an exported model: ``Predictor("runs/x/onnx").predict(["text", ...])``."""

    def __init__(
        self, path: str | Path, int8: bool | None = None, threads: int | None = None, max_length: int = 128
    ):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        path = Path(path)
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
        z = self.logits(texts, batch_size)
        z = np.exp(z - z.max(-1, keepdims=True))
        return z / z.sum(-1, keepdims=True)

    def predict(self, texts: list[str], batch_size: int = 32) -> list[str]:
        return [self.labels[i] for i in self.logits(texts, batch_size).argmax(-1)]
