"""Student: a small encoder classifier trained on the teacher's soft targets."""

from __future__ import annotations

import json
import math
import random
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .static import encoder_of, new_static  # also registers the static student with transformers

DEFAULT_STUDENT = "sentence-transformers/all-MiniLM-L6-v2"
TRIM_KEEP = 1000  # `train --trim-keep`: the default safety margin of trim_vocab
# common English words, for language_hint
ENGLISH_WORDS = frozenset(
    "the a an to of and is in i you my for it on what how can this that with be me do are".split()
    + "was have not your".split()
)


def language_hint(tok, texts: Sequence[str], n: int = 500) -> str | None:
    """A note when the texts look like another language than the student's tokenizer was made for: under 8%
    of their words are common English ones and the tokenizer needs over 1.5 tokens per word. On the example
    tasks' first 500 training texts with all-MiniLM-L6-v2, the English ones have 11% to 47% and 1.01 to 1.34,
    Danish MASSIVE 6% and 1.89; a multilingual tokenizer needs fewer tokens per word there, so its student
    gets no note."""
    words = [w for t in texts[:n] for w in re.findall(r"[^\W\d_]+", t.lower())]
    if len(words) < 50:
        return None
    english = sum(w in ENGLISH_WORDS for w in words) / len(words)
    pieces = sum(len(tok.tokenize(w)) for w in words) / len(words)
    if english >= 0.08 or pieces <= 1.5:
        return None
    return (
        f"the texts do not look English ({english:.0%} common English words, {pieces:.1f} of this student's "
        "tokens per word): a multilingual student, such as "
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 with --trim-vocab, is likely more "
        "accurate"
    )


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


def contrastive_pairs(classes: Sequence[int | None], n: int, seed: int = 0) -> list[tuple[int, int, float]]:
    """``n`` contrastive pairs of the texts that have a class (``classes[i]``, None for none), as SetFit
    generates them (Tunstall et al., 2022): ``n // 2`` positives, two different texts of one class with target
    1, and the rest negatives, texts of two different classes with target 0, in a seeded random order. A
    positive's first text is drawn from the texts whose class has another text, a negative's from all texts
    with a class; each partner is drawn from the texts that fit."""
    by_class: dict[int, list[int]] = {}
    for i, c in enumerate(classes):
        if c is not None:
            by_class.setdefault(c, []).append(i)
    anchors = [i for idx in by_class.values() if len(idx) > 1 for i in idx]
    if len(by_class) < 2 or not anchors:
        raise ValueError(
            f"contrastive pairs need texts of at least two classes and a class with two texts: "
            f"{sum(map(len, by_class.values()))} texts in {len(by_class)} classes"
        )
    labelled = [i for idx in by_class.values() for i in idx]
    rng = random.Random(seed)
    pairs: list[tuple[int, int, float]] = []
    for _ in range(n // 2):
        i = rng.choice(anchors)
        j = rng.choice(by_class[classes[i]])  # type: ignore[index]
        while j == i:
            j = rng.choice(by_class[classes[i]])  # type: ignore[index]
        pairs.append((i, j, 1.0))
    for _ in range(n - n // 2):
        i, j = rng.choice(labelled), rng.choice(labelled)
        while classes[j] == classes[i]:
            j = rng.choice(labelled)
        pairs.append((i, j, 0.0))
    rng.shuffle(pairs)
    return pairs


def _schedule(opt: torch.optim.Optimizer, steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up over the first tenth of the steps, then linear decay to zero."""
    warm = max(1, steps // 10)
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm + 1e-9)))
    )


def contrastive_warmup(
    model,
    tok,
    texts: Sequence[str],
    pairs: list[tuple[int, int, float]],
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-4,
    max_length: int = 128,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> None:
    """SetFit's contrastive step (Tunstall et al., 2022) on the student: the mean squared error between the
    cosine similarity of each pair's two representations (:func:`head_input`, what the classifier reads) and
    its target (1 same class, 0 different), for ``epochs`` passes over ``pairs`` in batches of
    ``batch_size`` pairs. The classifier's final layer gets no gradient and stays as it was."""
    rng = random.Random(seed)
    pairs = list(pairs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = _schedule(opt, epochs * math.ceil(len(pairs) / batch_size))
    model.train()
    for epoch in range(epochs):
        rng.shuffle(pairs)
        t0, total = time.perf_counter(), 0.0
        for b in range(0, len(pairs), batch_size):
            batch = pairs[b : b + batch_size]
            ids = sorted({i for i, j, _ in batch} | {j for i, j, _ in batch})
            at = {i: k for k, i in enumerate(ids)}
            f = head_input(model, tok, [texts[i] for i in ids], max_length).float()
            sim = F.cosine_similarity(f[[at[i] for i, _, _ in batch]], f[[at[j] for _, j, _ in batch]])
            loss = F.mse_loss(sim, torch.tensor([y for *_, y in batch], device=sim.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            total += loss.item() * len(batch)
        loss_ = total / len(pairs)
        log(f"warm-up epoch {epoch + 1}/{epochs}: cosine loss {loss_:.4f} ({time.perf_counter() - t0:.0f}s)")


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
    init_head: list[str] | None = None,
    outside: str | None = None,
    trim_texts: Sequence[str] | None = None,
    trim_keep: int = TRIM_KEEP,
    warmup_pairs: int = 0,
    warmup_epochs: int = 1,
    warmup_confidence: float = 0.9,
) -> int:
    """Train and save the student; returns the epoch whose model is saved. With ``val_share`` > 0 that share
    of the texts is held out (texts without gold labels when there is a teacher), the loss on them is measured
    after every epoch, the epoch with the lowest one is kept, and training stops after ``patience`` epochs
    without improvement. Otherwise every epoch is used and the last one is kept. ``chunk_tokens`` and
    ``chunk_stride`` (the texts are windows of long documents, see :func:`windows`) are saved with the model,
    which then reads every input in such windows; ``max_length`` is raised to fit a window if needed.
    ``init_head``: one text per label (its name, and description), to start the classifier from them with
    :func:`init_head_from_names`. ``outside``: the id of the task's `outside` label, saved with the model so
    that Predictor and the web page can tell it apart. ``trim_texts``: before training, cut the vocabulary to
    the tokens of these texts with :func:`trim_vocab` (and ``trim_keep``), so the student trains and is saved
    trimmed. ``warmup_pairs`` > 0: before distillation (and before ``init_head``), train the encoder with
    :func:`contrastive_warmup` for ``warmup_epochs`` on that many :func:`contrastive_pairs` of the training
    texts (not the held-out ones) with a class: the gold label where there is one, else the teacher's top
    label where its probability is at least ``warmup_confidence``. Single-label tasks only."""
    if warmup_pairs < 0 or (warmup_pairs and multi_label):
        raise ValueError(f"warmup_pairs must be 0 or more, and 0 for a multi-label task: {warmup_pairs}")
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
        if hint := language_hint(tok, texts):
            log(hint)
    if trim_texts is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        tok, before = trim_vocab(model, tok, trim_texts, out_dir, trim_keep)
        n = model.config.vocab_size
        log(f"vocabulary trimmed to {n:,} of {before:,} tokens, from {len(trim_texts)} texts")
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
    if warmup_pairs:
        sure: list[int | None] = [None] * len(texts)  # the teacher's top label where it is confident enough
        if teacher_probs is not None:
            p, top = teacher_probs.max(-1)
            sure = [
                int(c) if q >= warmup_confidence else None
                for q, c in zip(p.tolist(), top.tolist(), strict=True)
            ]
        classes = [
            None if i in held else s if g is None else g
            for i, (g, s) in enumerate(zip(gold, sure, strict=True))
        ]
        n_gold = sum(g is not None and i not in held for i, g in enumerate(gold))
        n = sum(c is not None for c in classes)
        log(f"contrastive warm-up: {warmup_pairs} pairs of {n} texts, {n_gold} with gold labels")
        pairs = contrastive_pairs(classes, warmup_pairs, seed + 3)  # type: ignore[arg-type]  # single-label
        contrastive_warmup(model, tok, texts, pairs, warmup_epochs, batch_size, lr, max_length, seed + 3, log)
    if init_head:
        init_head_from_names(model, tok, init_head, max_length=max_length)
        log("classifier initialised from the label names")

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
    sched = _schedule(opt, steps)

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
    if outside:  # read back by Predictor and the web pages
        model.config.tinydistill_outside = outside
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    kept = best_epoch if best_state is not None else len(history)
    (out_dir / "history.json").write_text(
        json.dumps({"epochs": history, "kept_epoch": kept, "held_out_texts": len(val)}, indent=1),
        encoding="utf-8",
    )
    return kept


def _remap_ids(node, remap: dict[int, int]) -> None:
    """Remap the token ids in a tokenizer.json post-processor: TemplateProcessing's ``ids`` lists and
    Bert/RobertaProcessing's ``[token, id]`` pairs, also inside a Sequence of processors."""
    if isinstance(node, list):
        for v in node:
            _remap_ids(v, remap)
    elif isinstance(node, dict):
        for k, v in node.items():
            if k == "ids":
                node[k] = [remap[i] for i in v]
            elif k in ("sep", "cls") and isinstance(v, list) and len(v) == 2 and isinstance(v[1], int):
                v[1] = remap[v[1]]
            else:
                _remap_ids(v, remap)


@torch.no_grad()
def trim_vocab(model, tok, texts: Sequence[str], out_dir: Path, keep_top: int = TRIM_KEEP):
    """Vocabulary trimming (Ushio et al., 2023, arXiv:2305.15020): keep only the embedding rows of the token
    ids ``tok`` produces on ``texts``, of the tokenizer's added and special tokens, of the token ids in the
    model's config, and of the ``keep_top`` most frequent pieces of the original vocabulary (Unigram: the
    highest scores; WordPiece and BPE: the lowest ids, which follow frequency or merge order), plus what the
    tokenizer needs to stay whole: a BPE token's merge parts, the byte tokens of a byte-fallback BPE and the
    256 characters of a byte-level one. The model's embedding matrix is cut to the kept rows in place, and the
    tokenizer, rebuilt so that its ids index them, is written to ``out_dir`` and returned with the original
    number of rows.

    The kept tokens keep their order, so a text whose tokens were all kept gets the same tokens as before and
    the same logits (WordPiece's longest match, Unigram's best segmentation and BPE's merges cannot pick a
    token that was not in the original result). A text with a trimmed token is split into kept pieces where it
    can be, else that part becomes the unknown token."""
    import re

    tok.save_pretrained(out_dir)
    file = out_dir / "tokenizer.json"
    if not file.exists():
        raise ValueError(
            f"trimming the vocabulary needs a fast tokenizer (tokenizer.json): {type(tok).__name__}"
        )
    spec = json.loads(file.read_text(encoding="utf-8"))
    m, kind = spec["model"], spec["model"]["type"]
    if kind not in ("WordPiece", "Unigram", "BPE"):
        raise ValueError(
            f"trimming the vocabulary supports WordPiece, Unigram and BPE tokenizers, not {kind}"
        )
    unigram = kind == "Unigram"
    vocab: dict[str, int] = {p: i for i, (p, _) in enumerate(m["vocab"])} if unigram else m["vocab"]
    added = {a["id"] for a in spec["added_tokens"]}
    emb = model.get_input_embeddings()
    rows = emb.num_embeddings

    keep = {i for ids in tok(list(texts), verbose=False)["input_ids"] for i in ids}
    keep |= added | set(tok.all_special_ids)
    keep |= {v for k, v in model.config.to_dict().items() if k.endswith("_token_id") and isinstance(v, int)}
    keep |= {vocab[m["unk_token"]]} if m.get("unk_token") in vocab else set()
    keep |= {m["unk_id"]} if unigram and m.get("unk_id") is not None else set()
    order = (
        sorted(range(len(m["vocab"])), key=lambda i: -m["vocab"][i][1]) if unigram else sorted(vocab.values())
    )
    names = {i: p for p, i in vocab.items()}
    # BERT's [unusedN] placeholders come first in its vocabulary but are never produced
    common = [i for i in order if i not in added and not re.fullmatch(r"\[unused\d+\]", names[i])]
    keep |= set(common[:keep_top])
    if m.get("byte_fallback"):  # <0x00>..<0xFF>: any text can still be written in bytes
        keep |= {i for p, i in vocab.items() if re.fullmatch(r"<0x[0-9A-F]{2}>", p)}
    if "ByteLevel" in json.dumps(spec.get("pre_tokenizer")):  # byte-level BPE has no unknown token
        from tokenizers.pre_tokenizers import ByteLevel

        keep |= {vocab[c] for c in ByteLevel.alphabet() if c in vocab}
    keep = {i for i in keep if i < rows}

    if kind == "BPE":  # every token a kept token is merged from, so the same merges still apply
        pre = m.get("continuing_subword_prefix") or ""
        pairs = [mg.split(" ", 1) if isinstance(mg, str) else mg for mg in m["merges"]]
        made = [a + (b[len(pre) :] if pre and b.startswith(pre) else b) for a, b in pairs]
        kept = {names[i] for i in keep if i in names}
        grown = True
        while grown:  # parts come before their merges, so one pass from the end nearly always suffices
            n = len(kept)
            for (a, b), r in zip(reversed(pairs), reversed(made), strict=True):
                if r in kept:
                    kept |= {a, b} & vocab.keys()
            grown = len(kept) > n
        keep |= {vocab[p] for p in kept}
        m["merges"] = [
            mg for mg, (a, b), r in zip(m["merges"], pairs, made, strict=True) if {a, b, r} <= kept
        ]

    new = sorted(keep)
    remap = {old: i for i, old in enumerate(new)}
    if unigram:  # kept pieces stay first and in order: model ids < len(vocab) <= added-only ids
        m["vocab"] = [m["vocab"][i] for i in new if i < len(m["vocab"])]
        m["unk_id"] = remap.get(m.get("unk_id"))
    else:
        m["vocab"] = {p: remap[i] for p, i in vocab.items() if i in remap}
    for a in spec["added_tokens"]:
        a["id"] = remap[a["id"]]
    _remap_ids(spec.get("post_processor"), remap)
    if spec.get("padding"):
        spec["padding"]["pad_id"] = remap[spec["padding"]["pad_id"]]
    file.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    config_file = out_dir / "tokenizer_config.json"  # transformers 4 lists the added tokens by id
    config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.exists() else {}
    if "added_tokens_decoder" in config:
        config["added_tokens_decoder"] = {
            str(remap[int(i)]): v for i, v in config["added_tokens_decoder"].items()
        }
        config_file.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    for name in tok.vocab_files_names.values():  # a slow tokenizer's files would describe the old ids
        if name != "tokenizer.json" and (out_dir / name).exists():
            if name == "vocab.txt" and kind == "WordPiece":
                ordered = sorted(m["vocab"], key=m["vocab"].get)
                (out_dir / name).write_text("".join(p + "\n" for p in ordered), encoding="utf-8")
            else:
                (out_dir / name).unlink()

    pos = getattr(getattr(model.base_model, "embeddings", None), "padding_idx", None)
    if isinstance(pos, int) and remap.get(pos) != pos:  # RoBERTa counts positions from the padding id
        raise ValueError(f"trimming would move the padding id {pos}, from which this model counts positions")
    pad = remap.get(emb.padding_idx) if emb.padding_idx is not None else None
    model.set_input_embeddings(
        torch.nn.Embedding.from_pretrained(emb.weight[new].clone(), freeze=False, padding_idx=pad)
    )
    model.config.vocab_size = len(new)
    for k, v in model.config.to_dict().items():
        if k.endswith("_token_id") and isinstance(v, int) and v in remap:
            setattr(model.config, k, remap[v])
    return AutoTokenizer.from_pretrained(out_dir), rows


@torch.no_grad()
def init_head_from_names(model, tok, names: list[str], scale: float = 10.0, max_length: int = 128) -> None:
    """Label-aware initialisation of the classifier's final linear layer: row c becomes the student's own
    representation of ``names[c]``, the input that layer receives for that text (the pooled sentence of a
    transformer student, the mean token vector of the static one), length-normalised and divided by the
    names' mean norm times ``scale``; the bias is zero. A text then scores each label about ``scale`` times
    its cosine similarity to the label's name, so the untrained student already ranks labels by it."""
    n = model.config.num_labels
    found = [m for m in model.modules() if isinstance(m, torch.nn.Linear) and m.out_features == n]
    if len(names) != n or not found:
        raise ValueError(
            f"cannot initialise the head: {len(names)} names, {n} labels, linear head found: {bool(found)}"
        )
    head = found[-1]  # the classifier is the last module registered (BERT, RoBERTa/MPNet, ModernBERT, static)
    was_training = model.training
    model.eval()
    try:
        f = head_input(model, tok, names, max_length).float()
    finally:
        model.train(was_training)
    head.weight.copy_(F.normalize(f, dim=-1) * (scale / f.norm(dim=-1).mean().clamp_min(1e-9)))
    if head.bias is not None:
        head.bias.zero_()


def head_input(model, tok, texts: Sequence[str], max_length: int = 128) -> torch.Tensor:
    """What the classifier's final linear layer receives for each text, one row per text: the pooled
    sentence of a transformer student, the mean token vector of the static one. Differentiable."""
    n = model.config.num_labels
    head = [m for m in model.modules() if isinstance(m, torch.nn.Linear) and m.out_features == n][-1]
    seen: list[torch.Tensor] = []
    hook = head.register_forward_hook(lambda _m, args, _out: seen.append(args[0]))
    try:
        enc = tok(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        model(**{k: v.to(head.weight.device) for k, v in enc.items()})
    finally:
        hook.remove()
    return seen[0]


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
