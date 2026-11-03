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


def distillation_loss_multilabel(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    gold: torch.Tensor | None = None,
    has_gold: torch.Tensor | None = None,
    ce_weight: float = 0.0,
) -> torch.Tensor:
    """Multi-label distillation: binary cross-entropy of each label's logit against the teacher's probability
    (equal to the per-label KL up to the teacher's entropy), averaged over labels and texts, plus ce_weight
    times binary cross-entropy against the gold labels (multi-hot) of the texts that have them."""
    z = student_logits.float()
    loss = F.binary_cross_entropy_with_logits(z, teacher_probs.float())
    if ce_weight and gold is not None and has_gold is not None and has_gold.any():
        loss = loss + ce_weight * F.binary_cross_entropy_with_logits(z[has_gold], gold[has_gold].float())
    return loss


def few_gold_sets(gold: list[list[int] | None], k: int, seed: int = 0) -> list[list[int] | None]:
    """Multi-label few_gold: for every label, the gold labels of up to k random texts that have it."""
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for i, g in enumerate(gold):
        for c in g or []:
            by_label.setdefault(c, []).append(i)
    keep: set[int] = set()
    for c in sorted(by_label):
        keep.update(rng.sample(by_label[c], min(k, len(by_label[c]))))
    return [g if i in keep else None for i, g in enumerate(gold)]


def few_shot_examples(
    texts: list[str], gold: list[int | None], k: int, seed: int = 0
) -> list[tuple[str, int]]:
    """The same K examples per class as few_gold, as (text, label) pairs in a seeded random order."""
    picked = [(t, g) for t, g in zip(texts, few_gold(gold, k, seed), strict=True) if g is not None]
    random.Random(seed).shuffle(picked)
    return picked


def few_gold(gold: list[int | None], k: int, seed: int = 0) -> list[int | None]:
    """Keep the gold label of k randomly chosen examples per class (fewer if a class has fewer) and drop the
    rest, so only a few labels per class are used."""
    by_class: dict[int, list[int]] = {}
    for i, g in enumerate(gold):
        if g is not None:
            by_class.setdefault(g, []).append(i)
    rng = random.Random(seed)
    keep: set[int] = set()
    for c in sorted(by_class):
        idx = by_class[c]
        keep.update(rng.sample(idx, min(k, len(idx))))
    return [g if i in keep else None for i, g in enumerate(gold)]


def train_student(
    texts: list[str],
    teacher_probs: torch.Tensor | None,
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
    multi_label: bool = False,
    on_step: Callable[[], None] | None = None,
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
        problem_type="multi_label_classification" if multi_label else "single_label_classification",
    ).to(device)
    gold = gold or [None] * len(texts)
    if multi_label:  # multi-hot gold, and which texts have it
        has_gold = torch.tensor([g is not None for g in gold])
        gold_t = torch.zeros(len(texts), len(labels))
        for i, g in enumerate(gold):
            gold_t[i, g or []] = 1.0
    else:
        gold_t = torch.tensor([-1 if g is None else g for g in gold])

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
            if multi_label:
                g = gold_t[idx].to(device)
                if teacher_probs is None:  # supervised only: every text has gold labels
                    loss = F.binary_cross_entropy_with_logits(logits.float(), g)
                else:
                    loss = distillation_loss_multilabel(
                        logits, teacher_probs[idx].to(device), g, has_gold[idx].to(device), ce_weight
                    )
            elif (
                teacher_probs is None
            ):  # supervised only (the upper-bound model): every text has a gold label
                loss = F.cross_entropy(logits.float(), gold_t[idx].to(device))
            else:
                loss = distillation_loss(
                    logits, teacher_probs[idx].to(device), gold_t[idx].to(device), ce_weight
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            total += loss.item() * len(idx)
            if on_step:
                on_step()
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
