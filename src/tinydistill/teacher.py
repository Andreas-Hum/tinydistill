"""Teacher: score each label's verbalisation under a causal LM, cache the scores on disk.

For a text x and a label with continuation tokens c_1..c_m (the verbalisation followed by the end-of-turn
token), the score is   log p(c_1..c_m | prompt(x)) = sum_j log p(c_j | prompt(x), c_<j).
A softmax over the label scores gives the teacher's soft targets. There is no free-form generation.

Speed: the prompt is split around the text into a constant prefix and a per-text suffix. The prefix's KV
cache is computed once. Per text, one forward pass covers the text and all labels at once: the labels' tokens
form a prefix tree (labels that start alike share nodes), and a 4D attention mask lets every tree node attend
to the cached prompt, the text and its own ancestors only. There is no padding and no per-label copy of the
cache; the cache is cropped back to the prompt prefix afterwards.
"""

from __future__ import annotations

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
    def __init__(self, model, tokenizer, task: Task, use_chat_template: bool = True):
        self.model = model.eval()
        self.tok = tokenizer
        self.task = task
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
        self._build_tree()
        self._prefix_cache = None

    def _build_tree(self) -> None:
        """Prefix tree over the labels' input tokens (every label token except the last).

        Node i stands for one distinct label prefix (c_1..c_d); its input token is c_d and its output predicts
        c_(d+1). The first label token is predicted by the last text position (row -1 below).
        """
        nodes: dict[tuple[int, ...], int] = {}
        parent: list[int] = []
        depth: list[int] = []
        tokens: list[int] = []
        rows: list[int] = []  # for every label token, which output row predicts it (-1 = last text position)
        targets: list[int] = []
        owner: list[int] = []
        for li, lab in enumerate(self.label_ids):
            for j, tok in enumerate(lab):
                rows.append(nodes[tuple(lab[:j])] if j else -1)
                targets.append(tok)
                owner.append(li)
                if j < len(lab) - 1 and tuple(lab[: j + 1]) not in nodes:
                    nodes[tuple(lab[: j + 1])] = len(tokens)
                    parent.append(nodes[tuple(lab[:j])] if j else -1)
                    depth.append(j)
                    tokens.append(tok)
        n = len(tokens)
        anc = torch.eye(n, dtype=torch.bool)
        for i in range(n):  # parents always come before children
            if parent[i] >= 0:
                anc[i] |= anc[parent[i]]
        dev = self.device
        self._node_tokens = torch.tensor(tokens, dtype=torch.long, device=dev)
        self._node_depth = torch.tensor(depth, dtype=torch.long, device=dev)
        self._node_anc = anc.to(dev)
        self._rows = torch.tensor(rows, dtype=torch.long, device=dev)
        self._targets = torch.tensor(targets, dtype=torch.long, device=dev)
        self._owner = torch.tensor(owner, dtype=torch.long, device=dev)

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
        P = len(self.prefix_ids)
        if self._prefix_cache is None:
            out = self.model(torch.tensor([self.prefix_ids], device=dev), use_cache=True)
            self._prefix_cache = out.past_key_values
            self._prefix_last = torch.log_softmax(out.logits[0, -1].float(), -1)
        ids = torch.tensor(self.text_ids(text), dtype=torch.long, device=dev)
        n, N = len(ids), len(self._node_tokens)
        if n == 0 and N == 0:
            return self._prefix_last[self._targets].tolist()

        # Query rows: n text tokens, then N tree nodes. Keys: P cached prompt tokens, then the same rows.
        allow = torch.zeros((n + N, P + n + N), dtype=torch.bool, device=dev)
        allow[:, :P] = True
        allow[:n, P : P + n] = torch.ones((n, n), dtype=torch.bool, device=dev).tril()
        allow[n:, P : P + n] = True
        allow[n:, P + n :] = self._node_anc
        positions = torch.cat([torch.arange(P, P + n, device=dev), P + n + self._node_depth])
        keep = torch.arange(max(n - 1, 0), n + N, device=dev)  # last text position + every node
        try:
            out = self.model(
                torch.cat([ids, self._node_tokens])[None],
                attention_mask=allow[None, None],
                position_ids=positions[None],
                past_key_values=self._prefix_cache,
                use_cache=True,
                logits_to_keep=keep,
            )
        finally:
            extra = self._prefix_cache.get_seq_length() - P
            if extra:
                self._prefix_cache.crop(-extra)  # back to the prompt prefix
        lp = torch.log_softmax(out.logits[0].float(), -1)  # row 0: last text position, row 1 + i: node i
        if n == 0:  # the prompt prefix itself predicts the first label token
            lp = torch.cat([self._prefix_last[None], lp])
        per_token = lp[self._rows + 1, self._targets]
        total = torch.zeros(len(self.label_ids), dtype=per_token.dtype, device=dev)
        return total.index_add_(0, self._owner, per_token).tolist()


def load_teacher(name: str, task: Task, device: str = "auto", load_4bit: bool = False) -> LabelScorer:
    device = pick_device(device)
    tok = AutoTokenizer.from_pretrained(name)
    kwargs: dict = {}
    if load_4bit:
        if not device.startswith("cuda"):
            raise SystemExit("--load-4bit needs a CUDA device (bitsandbytes); use --device cuda")
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
    return LabelScorer(model, tok, task)


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
