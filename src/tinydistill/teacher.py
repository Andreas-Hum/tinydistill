"""Teacher: score each label's verbalisation under a causal LM, cache the scores on disk.

For a text x and a label with continuation tokens c_1..c_m (the verbalisation followed by the end-of-turn
token), the score is   log p(c_1..c_m | prompt(x)) = sum_j log p(c_j | prompt(x), c_<j).
A softmax over the label scores gives the teacher's soft targets. There is no free-form generation.

Speed: the prompt is split around the text into a constant prefix and a per-text suffix. The prefix's KV
cache is computed once, extended with each text, then fanned out over the labels in batches.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .task import Task

_SENTINEL = "\x00TEXT\x00"


def pick_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


class LabelScorer:
    def __init__(self, model, tokenizer, task: Task, label_batch: int = 32, use_chat_template: bool = True):
        self.model = model.eval()
        self.tok = tokenizer
        self.task = task
        self.label_batch = label_batch
        self.device = next(model.parameters()).device
        self.chat = use_chat_template and bool(getattr(tokenizer, "chat_template", None))

        # Split the rendered prompt around the text.
        user = task.render(_SENTINEL)
        if self.chat:
            full = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
            )
        else:
            full = user
        prefix, suffix = full.split(_SENTINEL)
        self.suffix = suffix
        ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        if not self.chat and tokenizer.bos_token_id is not None:
            ids = [tokenizer.bos_token_id, *ids]
        self.prefix_ids: list[int] = ids

        # Continuations: verbalisation + end-of-turn token. Without a chat template the answer follows the
        # prompt text directly, so it starts with a space.
        end = tokenizer.eos_token_id
        self.label_ids: list[list[int]] = []
        for v in task.verbalisations:
            t = tokenizer(v if self.chat else " " + v, add_special_tokens=False)["input_ids"]
            self.label_ids.append(t + ([end] if end is not None else []))
        self._prefix_cache = None

    def text_ids(self, text: str) -> list[int]:
        return self.tok(text + self.suffix, add_special_tokens=False)["input_ids"]

    def fingerprint(self) -> str:
        """Identifies everything that changes the scores, so caches are never mixed up."""
        cfg = getattr(self.model, "config", None)
        key = {
            "model": getattr(cfg, "_name_or_path", ""),
            "dtype": str(next(self.model.parameters()).dtype),
            "quant": str(getattr(cfg, "quantization_config", "")),
            "prefix": self.prefix_ids,
            "suffix": self.suffix,
            "labels": self.label_ids,
        }
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]

    @torch.no_grad()
    def score(self, text: str) -> list[float]:
        """Summed log-probability of every label's continuation given the prompt for ``text``."""
        dev = self.device
        if self._prefix_cache is None:
            out = self.model(torch.tensor([self.prefix_ids], device=dev), use_cache=True)
            self._prefix_cache = out.past_key_values
            self._prefix_last = out.logits[0, -1]
        cache = copy.deepcopy(self._prefix_cache)
        ids = self.text_ids(text)
        if ids:
            out = self.model(torch.tensor([ids], device=dev), past_key_values=cache, use_cache=True)
            first = out.logits[0, -1]
        else:
            first = self._prefix_last
        first_lp = torch.log_softmax(first.float(), -1)

        scores: list[float] = []
        for start in range(0, len(self.label_ids), self.label_batch):
            chunk = self.label_ids[start : start + self.label_batch]
            m = max(len(c) for c in chunk)
            tokens = torch.full((len(chunk), m), self.tok.pad_token_id or 0, dtype=torch.long, device=dev)
            mask = torch.zeros((len(chunk), m), dtype=torch.bool, device=dev)
            for i, c in enumerate(chunk):
                tokens[i, : len(c)] = torch.tensor(c, device=dev)
                mask[i, : len(c)] = True
            total = first_lp[tokens[:, 0]]
            if m > 1:
                # Right padding is safe: causal attention means pads only affect positions after them.
                c2 = copy.deepcopy(cache)
                c2.batch_repeat_interleave(len(chunk))
                logits = self.model(tokens[:, :-1], past_key_values=c2, use_cache=True).logits
                lp = torch.log_softmax(logits.float(), -1).gather(-1, tokens[:, 1:, None])[..., 0]
                total = total + (lp * mask[:, 1:]).sum(-1)
            scores.extend(total.tolist())
        return scores


def load_teacher(
    name: str, task: Task, device: str = "auto", load_4bit: bool = False, label_batch: int = 32
) -> LabelScorer:
    device = pick_device(device)
    tok = AutoTokenizer.from_pretrained(name)
    kwargs: dict = {}
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
        kwargs["device_map"] = {"": device}
    else:
        kwargs["dtype"] = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if not load_4bit:
        model.to(device)
    return LabelScorer(model, tok, task, label_batch=label_batch)


class ScoreCache:
    """Append-only JSONL of {"h": text hash, "s": label scores}; reopening it resumes where it stopped."""

    def __init__(self, path: Path):
        self.path = path
        self.scores: dict[str, list[float]] = {}
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:  # a line cut off by an interrupted run
                        continue
                    self.scores[row["h"]] = row["s"]

    def fill(self, scorer: LabelScorer, texts: Iterable[str], progress=None) -> int:
        """Score every text not in the cache yet. Returns how many were scored now."""
        n = 0
        with self.path.open("a", encoding="utf-8") as f:
            for text in texts:
                h = text_hash(text)
                if h in self.scores:
                    continue
                s = scorer.score(text)
                self.scores[h] = s
                f.write(json.dumps({"h": h, "s": s}) + "\n")
                f.flush()
                n += 1
                if progress:
                    progress(n)
        return n

    def get(self, text: str) -> list[float]:
        return self.scores[text_hash(text)]


def iter_unique(texts: Iterable[str]) -> Iterator[str]:
    seen: set[str] = set()
    for t in texts:
        if t not in seen:
            seen.add(t)
            yield t
