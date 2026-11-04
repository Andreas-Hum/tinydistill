"""Student: a small encoder classifier trained on the teacher's soft targets."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .static import encoder_of, new_static  # also registers the static student with transformers

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
    gold: Sequence[int | list[int] | None] | None = None,
    ce_weight: float = 0.0,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    max_length: int = 128,
    device: str = "cpu",
    seed: int = 0,
    log: Callable[[str], None] = print,
    multi_label: bool = False,
    on_step: Callable[[], None] | None = None,
    val_share: float = 0.0,
    patience: int = 3,
    chunk_tokens: int | None = None,
    chunk_stride: int | None = None,
) -> int:
    """Train and save the student; returns the epoch whose model is saved. With ``val_share`` > 0 that share
    of the texts is held out (texts without gold labels when there is a teacher), the loss on them is measured
    after every epoch, the epoch with the lowest one is kept, and training stops after ``patience`` epochs
    without improvement. Otherwise every epoch is used and the last one is kept. ``chunk_tokens`` and
    ``chunk_stride`` (the texts are windows of long documents, see :func:`windows`) are saved with the model,
    which then reads every input in such windows; ``max_length`` is raised to fit a window if needed."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    if static := encoder_of(model_name):
        model, tok = new_static(static, labels, multi_label)
    else:
        from transformers.utils import logging as hf_logging

        tok = AutoTokenizer.from_pretrained(model_name)
        level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()  # its report of the new (missing) classifier head is expected noise
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(labels),
                id2label=dict(enumerate(labels)),
                label2id={name: i for i, name in enumerate(labels)},
                ignore_mismatched_sizes=True,
                problem_type="multi_label_classification" if multi_label else "single_label_classification",
            )
        finally:
            hf_logging.set_verbosity(level)
        log(f"{model_name}: a new classification head for {len(labels)} labels on the pretrained encoder")
    model = model.to(device)
    if chunk_tokens and max_length < chunk_tokens + tok.num_special_tokens_to_add():
        max_length = chunk_tokens + tok.num_special_tokens_to_add()
        log(f"max_length raised to {max_length}: a window of {chunk_tokens} tokens plus special tokens")
    gold = gold or [None] * len(texts)
    if multi_label:  # multi-hot gold, and which texts have it
        has_gold = torch.tensor([g is not None for g in gold])
        gold_t = torch.zeros(len(texts), len(labels))
        for i, g in enumerate(gold):
            gold_t[i, g or []] = 1.0
    else:
        gold_t = torch.tensor([-1 if g is None else g for g in gold])

    val: list[int] = []
    if val_share > 0:
        pool = [i for i, g in enumerate(gold) if teacher_probs is None or g is None]
        val = sorted(
            random.Random(seed + 2).sample(pool, min(len(pool), max(1, round(val_share * len(texts)))))
        )
    held = set(val)
    order = [i for i in range(len(texts)) if i not in held]

    def loss_of(idx: list[int]) -> torch.Tensor:
        enc = tok(
            [texts[i] for i in idx], padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        if multi_label:
            g = gold_t[idx].to(device)
            if teacher_probs is None:  # supervised only: every text has gold labels
                return F.binary_cross_entropy_with_logits(logits.float(), g)
            return distillation_loss_multilabel(
                logits, teacher_probs[idx].to(device), g, has_gold[idx].to(device), ce_weight
            )
        if teacher_probs is None:  # supervised only (baseline, upper bound): every text has a gold label
            return F.cross_entropy(logits.float(), gold_t[idx].to(device))
        return distillation_loss(logits, teacher_probs[idx].to(device), gold_t[idx].to(device), ce_weight)

    steps = epochs * math.ceil(len(order) / batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm + 1e-9)))
    )

    best_loss, best_state, best_epoch = math.inf, None, epochs
    history: list[dict] = []  # per epoch, saved as history.json next to the model
    for epoch in range(epochs):
        model.train()
        rng.shuffle(order)
        t0, total = time.perf_counter(), 0.0
        for b in range(0, len(order), batch_size):
            idx = order[b : b + batch_size]
            loss = loss_of(idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            total += loss.item() * len(idx)
            if on_step:
                on_step()
        line = f"epoch {epoch + 1}/{epochs}: loss {total / len(order):.4f}"
        history.append({"epoch": epoch + 1, "loss": total / len(order)})
        if val:
            model.eval()
            with torch.no_grad():
                v = sum(
                    loss_of(val[b : b + 64]).item() * len(val[b : b + 64]) for b in range(0, len(val), 64)
                )
            v /= len(val)
            line += f", held-out loss {v:.4f}"
            history[-1]["held_out_loss"] = v
            if v < best_loss:
                best_loss, best_epoch = v, epoch + 1
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        log(f"{line} ({time.perf_counter() - t0:.0f}s)")
        if val and epoch + 1 - best_epoch >= patience:
            log(f"no improvement on the held-out texts for {patience} epochs: stopping")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        log(f"keeping epoch {best_epoch} (lowest held-out loss, on {len(val)} texts)")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.config.tinydistill_max_length = (
        max_length  # read back by student_logits, Predictor and the web page
    )
    if chunk_tokens:  # read back by student_logits and Predictor
        model.config.tinydistill_chunk_tokens = chunk_tokens
        model.config.tinydistill_chunk_stride = chunk_stride or max(1, chunk_tokens // 2)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    kept = best_epoch if best_state is not None else len(history)
    (out_dir / "history.json").write_text(
        json.dumps({"epochs": history, "kept_epoch": kept, "held_out_texts": len(val)}, indent=1),
        encoding="utf-8",
    )
    return kept


def load_student(path: Path, device: str = "cpu"):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
    return model, tok


@torch.no_grad()
def student_logits(
    model, tok, texts: list[str], batch_size: int = 64, max_length: int | None = None
) -> torch.Tensor:
    """Logits per text. Inputs are cut to ``max_length`` tokens, by default the length the student was trained
    with (128 for students trained before it was recorded). A student trained on windows of long documents
    reads each text in the same windows; see :func:`by_windows`."""
    max_length = max_length or getattr(model.config, "tinydistill_max_length", None) or 128
    dev = next(model.parameters()).device

    def run(texts: list[str]) -> torch.Tensor:
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

    size = getattr(model.config, "tinydistill_chunk_tokens", None)
    if not size or not texts:
        return run(texts)
    multi = model.config.problem_type == "multi_label_classification"
    stride = model.config.tinydistill_chunk_stride
    rows = by_windows(lambda w: run(w).tolist(), texts, lambda t: windows(tok, t, size, stride), multi)
    return torch.tensor(rows, dtype=torch.float32)


def windows(tok, text: str, size: int, stride: int | None = None) -> list[str]:
    """``text`` as overlapping windows of ``size`` tokens of ``tok`` (no special tokens), each starting
    ``stride`` tokens (default ``size // 2``) after the previous one, decoded back to text; the last one may
    be shorter. A text of at most ``size`` tokens is one window, the text itself."""
    stride = stride or max(1, size // 2)
    ids = tok(text, add_special_tokens=False, verbose=False)["input_ids"]
    if len(ids) <= size:
        return [text]
    return [tok.decode(ids[s : s + size]) for s in range(0, len(ids) - size + stride, stride)]


def by_windows(
    score: Callable[[list[str]], list[list[float]]],
    texts: list[str],
    cut: Callable[[str], list[str]],
    multi_label: bool = False,
) -> list[list[float]]:
    """Score every text's windows (``cut``) in one call of ``score`` and pool them into one row per text: the
    log of the mean of the windows' probabilities (softmax; multi-label, each label's sigmoid, as log-odds),
    as ``combine_scores`` pools teachers. A text that is a single window keeps its scores unchanged."""
    from .teacher import combine_scores

    parts = [cut(t) for t in texts]
    rows = score([w for p in parts for w in p])
    out, i = [], 0
    for p in parts:
        out.append(combine_scores(rows[i : i + len(p)], multi_label))
        i += len(p)
    return out
