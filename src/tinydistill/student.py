"""Student: a small encoder classifier trained on the teacher's soft targets."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_STUDENT = "sentence-transformers/all-MiniLM-L6-v2"


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    gold: torch.Tensor | None = None,
    ce_weight: float = 0.0,
) -> torch.Tensor:
    """Mean over the batch of KL(teacher || student), plus ce_weight * mean CE on the examples with a gold
    label (gold = -1 marks examples without one). Standard knowledge distillation (Hinton et al., 2015)."""
    log_q = F.log_softmax(student_logits.float(), dim=-1)
    p = teacher_probs.float()
    kl = (p * (torch.log(p.clamp_min(1e-12)) - log_q)).sum(-1).mean()
    if ce_weight and gold is not None and (gold >= 0).any():
        has = gold >= 0
        kl = kl + ce_weight * F.nll_loss(log_q[has], gold[has])
    return kl


def train_student(
    texts: list[str],
    teacher_probs: torch.Tensor,
    labels: list[str],
    out_dir: Path,
    model_name: str = DEFAULT_STUDENT,
    gold: list[int | None] | None = None,
    ce_weight: float = 0.0,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    max_length: int = 128,
    device: str = "cpu",
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> None:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={name: i for i, name in enumerate(labels)},
        ignore_mismatched_sizes=True,
    ).to(device)
    gold_t = torch.tensor([-1 if g is None else g for g in (gold or [None] * len(texts))])

    steps = epochs * math.ceil(len(texts) / batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm + 1e-9)))
    )

    order = list(range(len(texts)))
    for epoch in range(epochs):
        model.train()
        rng.shuffle(order)
        t0, total = time.perf_counter(), 0.0
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            enc = tok(
                [texts[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            loss = distillation_loss(logits, teacher_probs[idx].to(device), gold_t[idx].to(device), ce_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            total += loss.item() * len(idx)
        log(f"epoch {epoch + 1}/{epochs}: loss {total / len(texts):.4f} ({time.perf_counter() - t0:.0f}s)")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


def load_student(path: Path, device: str = "cpu"):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    return model, tok


@torch.no_grad()
def student_logits(model, tok, texts: list[str], batch_size: int = 64, max_length: int = 128) -> torch.Tensor:
    dev = next(model.parameters()).device
    out = []
    for b in range(0, len(texts), batch_size):
        enc = tok(
            texts[b : b + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out.append(model(**{k: v.to(dev) for k, v in enc.items()}).logits.float().cpu())
    return torch.cat(out) if out else torch.empty(0)
