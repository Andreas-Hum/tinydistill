"""Check encoders as students on a CPU: 20 training steps on a few texts (with --init-head names), ONNX fp32
and int8 export, the fp32 model against PyTorch, and the int8 model's size and latency with Predictor.
A smoke test of each architecture, not a quality measurement. Writes encoders.json next to this file.

    OMP_NUM_THREADS=2 python examples/students-cpu/measure.py [student ...]
"""

from __future__ import annotations

import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from tinydistill.evaluate import bench, dir_size_mb
from tinydistill.export import Predictor, export_onnx
from tinydistill.student import load_student, student_logits, train_student

STUDENTS = [
    "sentence-transformers/all-MiniLM-L6-v2",  # reference: the default student
    "jhu-clsp/ettin-encoder-17m",
    "jhu-clsp/ettin-encoder-32m",
    "jhu-clsp/mmBERT-small",
    "static:minishlab/potion-base-8M",
    "static:minishlab/potion-multilingual-128M",
]
LABELS = ["card", "transfer", "refund", "balance"]
TEXTS = [
    ("my card has not arrived yet, how long does delivery take", 0),
    ("I lost my card on the train, please block it", 0),
    ("can I order a second card for my partner", 0),
    ("the card machine rejected my card again today", 0),
    ("how do I send money to a friend abroad", 1),
    ("the transfer to my landlord is still pending", 1),
    ("why did my bank transfer fail this morning", 1),
    ("is there a fee for transfers between my own accounts", 1),
    ("the shop agreed to refund me but nothing has come back", 2),
    ("I returned the item two weeks ago and want my money", 2),
    ("how long does a refund usually take to show up", 2),
    ("a merchant charged me twice, can I get one back", 2),
    ("what is my current balance", 3),
    ("my balance did not update after the deposit", 3),
    ("why is my available balance lower than my balance", 3),
    ("show me the balance on my savings account", 3),
]
THREADS = 2


def check(student: str, work: Path) -> dict:
    texts = [t for t, _ in TEXTS]
    probs = torch.full((len(texts), len(LABELS)), 0.1 / 3)
    for i, (_, c) in enumerate(TEXTS):
        probs[i, c] = 0.9
    t0 = time.perf_counter()
    # 16 texts, batch 4, 5 epochs: 20 steps
    train_student(
        texts,
        probs,
        LABELS,
        work / "s",
        student,
        epochs=5,
        batch_size=4,
        lr=1e-4,
        init_head=LABELS,
        log=print,
    )
    train_s = time.perf_counter() - t0
    model, tok = load_student(work / "s")
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    want = student_logits(model, tok, texts).numpy()
    export_onnx(work / "s", work / "onnx", int8=True)
    fp32 = Predictor(work / "onnx", int8=False, threads=THREADS)
    int8 = Predictor(work / "onnx", int8=True, threads=THREADS)
    q = int8.logits(texts)
    row = {
        "student": student,
        "params_m": round(params_m, 1),
        "train_20_steps_s": round(train_s, 1),
        "onnx_fp32_max_abs_diff_vs_pytorch": float(np.abs(fp32.logits(texts) - want).max()),
        "int8_top_label_agrees_with_pytorch": float((q.argmax(-1) == want.argmax(-1)).mean()),
        "size_mb_fp32": round(dir_size_mb([work / "onnx" / "model.onnx"]), 1),
        "size_mb_int8": round(dir_size_mb([work / "onnx" / "model.int8.onnx"]), 1),
    }
    for name, pr in (("int8", int8), ("fp32", fp32)):
        b = bench(lambda x, pr=pr: pr.logits(x, batch_size=len(x)), texts, 50, 10)
        row[f"latency_ms_b1_{name}"] = round(b["latency_ms_b1"], 2)
        row[f"latency_ms_b32_{name}"] = round(b["latency_ms_b32"], 1)
    print(json.dumps(row))
    return row


def main() -> None:
    torch.set_num_threads(THREADS)
    out = Path(__file__).with_name("encoders.json")
    todo = sys.argv[1:] or STUDENTS
    rows = json.loads(out.read_text(encoding="utf-8"))["students"] if out.exists() else []
    rows = [r for r in rows if r["student"] not in todo]  # students given again are measured again
    for student in todo:
        with tempfile.TemporaryDirectory() as d:
            rows.append(check(student, Path(d)))
    import onnxruntime
    import transformers

    out.write_text(
        json.dumps(
            {
                "what": "20 training steps on 16 short texts (4 labels) on CPU, then ONNX export; latency "
                "is the median over 50 calls of Predictor.logits at batch 1 (10 at batch 32) on those texts",
                "threads": THREADS,
                "cpu": platform.processor() or platform.machine(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "onnxruntime": onnxruntime.__version__,
                "students": rows,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
