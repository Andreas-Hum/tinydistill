"""Teacher: score each label's verbalisation under a causal LM, cache the scores on disk.

For a text x and a continuation c_1..c_m (a way of writing a label, followed by the end-of-turn token),
log p(c | prompt(x)) = sum_j log p(c_j | prompt(x), c_<j). A label can be written in several surface forms
(lowercase, Capitalised, UPPERCASE, as written; each with and without a leading space), and its score is the
log of the summed probability of its distinct forms. A softmax over the label scores gives the teacher's soft
targets. There is no free-form generation, except for a teacher that reasons first (ReasoningScorer).

Speed: the prompt is split around the text into a constant prefix and a per-text suffix. The prefix's KV
cache is computed once. A text and all its continuations form one segment: the text's tokens, then a prefix
tree of the continuations' tokens (continuations that start alike share nodes). Several segments are packed
into one sequence, up to a token budget, and a 4D attention mask lets every position attend to the cached
prompt plus, within its own segment, the text (causally) and its own tree ancestors. One forward pass scores
the whole pack; there is no padding and no copy of the cache, which is cropped back to the prompt afterwards.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple, Protocol
from urllib.parse import urlsplit

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .task import LETTERS, Task


class Scorer(Protocol):
    """What every teacher (local, vLLM or hosted API) offers: label scores for texts."""

    task: Task

    def score_many(self, texts: list[str]) -> list[list[float]]: ...

    def score(self, text: str) -> list[float]: ...

    def fingerprint(self) -> str: ...


_SENTINEL = "\x00TEXT\x00"
_LABEL = "\x00LABEL\x00"
_EXAMPLES = "\x00EXAMPLES\x00"


def pick_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def surface_variants(name: str, kinds=("lowercase", "capitalised", "uppercase")) -> list[str]:
    """Common ways of writing a label: as written, plus the chosen kinds of lowercase, Capitalised and
    UPPERCASE; each with and without a leading space. The caller drops forms that tokenise identically."""
    low = name.lower()
    variant = {"lowercase": low, "capitalised": low[:1].upper() + low[1:], "uppercase": name.upper()}
    cased = [name, *(variant[k] for k in ("lowercase", "capitalised", "uppercase") if k in kinds)]
    out: list[str] = []
    for form in cased:
        for v in (form, " " + form):
            if v not in out:
                out.append(v)
    return out


def _group_logsumexp(f: torch.Tensor, group: torch.Tensor, n: int) -> torch.Tensor:
    """(rows, items) values -> (rows, n) log-sum-exp over the items of each group."""
    idx = group.expand_as(f)
    top = torch.full((f.shape[0], n), -torch.inf, dtype=f.dtype, device=f.device).scatter_reduce(
        1, idx, f, "amax"
    )
    total = torch.zeros_like(top).scatter_add_(1, idx, (f - top.gather(1, idx)).exp())
    return top + total.log()


def end_of_turn(tokenizer, chat: bool) -> list[int]:
    """The token that ends the assistant's answer. With a chat template it is read from the template (render
    an answer and take the first token after it): many models end a turn with a token that is not
    ``eos_token`` (Phi-3 <|end|>, Gemma <end_of_turn>, Llama 3 <|eot_id|>). Otherwise ``eos_token``."""
    if chat:
        marker = "\x00ANSWER\x00"
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": "q"}, {"role": "assistant", "content": marker}], tokenize=False
            )
        except Exception:  # templates that cannot render an assistant turn
            text = ""
        after = (text.split(marker, 1)[1] if marker in text else "").lstrip()  # Llama 2: " </s>"
        ids = tokenizer(after, add_special_tokens=False)["input_ids"] if after else []
        if ids:
            return ids[:1]
    return [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []


class _Prompt:
    """The rendered prompt split around the text, and every label's continuations, as token ids. Shared by the
    local scorer and the vLLM client, which score the same continuations.

    With ``similar`` (a ``Retriever``: each text's own few-shot examples) the prompt is split at {examples}
    instead: the cached prefix ends there, and a text's tokens are its examples, the prompt between
    {examples} and {text}, the text and the rest."""

    middle = ""  # with similar: the prompt between {examples} and {text}

    def _prepare(
        self,
        tokenizer,
        task: Task,
        use_chat_template: bool,
        surface_forms: bool,
        examples: list[tuple[str, int]] | None,
        max_positions: int | None,
        similar=None,
    ) -> None:
        self.tok = tokenizer
        self.task = task
        self.chat = use_chat_template and bool(getattr(tokenizer, "chat_template", None))

        # Split the rendered prompt around the text (and, multi-label, around the label being asked about).
        self.examples = examples or []
        self.similar = similar
        if similar is not None:
            at = task.prompt.find("{examples}")
            if at < 0 or at > task.prompt.find("{text}"):
                raise ValueError(
                    "similar few-shot examples need {examples} before {text} in the task's prompt"
                )
            task = replace(task, prompt=task.prompt.replace("{examples}", _EXAMPLES))
        user = task.render(_SENTINEL, self.examples, _LABEL if task.multi_label else None)
        if self.chat:
            full = tokenizer.apply_chat_template(  # no thinking: the answer must come right after the prompt
                [{"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Qwen3 style templates open a <think> block otherwise; others ignore
            )
        else:
            full = user
        prefix, suffix = full.split(_SENTINEL)
        if similar is not None:
            prefix, self.middle = prefix.split(_EXAMPLES)
        rest = ""  # multi-label: what follows {label}, up to where the answer starts
        if task.multi_label:
            suffix, rest = suffix.split(_LABEL)
        self.suffix = suffix
        ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        if not self.chat and tokenizer.bos_token_id is not None:
            ids = [tokenizer.bos_token_id, *ids]
        self.prefix_ids: list[int] = ids
        if isinstance(max_positions, int) and len(ids) >= max_positions:
            raise ValueError(
                f"the prompt has {len(ids)} tokens, more than the teacher's {max_positions}; "
                "use fewer examples"
            )

        # Continuations: each surface form of each label name + the end-of-turn token. Forms that tokenise
        # identically are kept once. Without surface_forms only the name as written is scored (with a leading
        # space when there is no chat template, since the answer then follows the prompt text directly).
        end = end_of_turn(tokenizer, self.chat)
        self.forms: list[list[int]] = []  # token ids of every continuation
        self.form_label: list[int] = []  # which label each continuation belongs to
        self.form_answer: list[int] = []  # multi-label: 0 = the positive answer, 1 = the negative one
        seen: dict[tuple[int, ...], int] = {}
        if task.multi_label:
            # Per label: the question's remainder ("{label}" onwards), then each surface form of each answer.
            # Both answers share the question tokens, so their probability cancels in the label's log-odds.
            for li, name in enumerate(task.verbalisations):
                question = tokenizer(name + rest, add_special_tokens=False)["input_ids"]
                for ai, answer in enumerate(task.answers):
                    for text in surface_variants(answer, task.surface_forms) if surface_forms else [answer]:
                        ids = tuple(question + tokenizer(text, add_special_tokens=False)["input_ids"] + end)
                        if ids not in seen:
                            seen[ids] = li
                            self.forms.append(list(ids))
                            self.form_label.append(li)
                            self.form_answer.append(ai)
            return
        for li, name in enumerate(task.verbalisations):
            texts = (
                surface_variants(name, task.surface_forms)
                if surface_forms
                else [name if self.chat else " " + name]
            )
            for text in texts:
                ids = tuple(tokenizer(text, add_special_tokens=False)["input_ids"] + end)
                if seen.get(ids, li) != li:
                    other = task.verbalisations[seen[ids]]
                    raise ValueError(f"labels {other!r} and {name!r} share the surface form {text!r}")
                if ids not in seen:
                    seen[ids] = li
                    self.forms.append(list(ids))
                    self.form_label.append(li)

    def text_ids(self, text: str, head: str = "") -> list[int]:
        return self.tok(head + text + self.suffix, add_special_tokens=False)["input_ids"]

    def heads(self, texts: list[str]) -> list[str]:
        """With ``similar``, per text: what comes between the cached prefix and the text (its examples and the
        prompt up to {text}, as ``Task.render`` writes them); otherwise nothing."""
        if self.similar is None:
            return [""] * len(texts)
        middle = self.middle.removeprefix("\n")  # no examples: render drops the {examples} line
        return [self.task.shots(ex) + self.middle if ex else middle for ex in self.similar(texts)]

    def _label_scores(self, f: torch.Tensor) -> list[list[float]]:
        """(texts, forms) log-probabilities -> per label, the log of the summed probability of its forms.
        Multi-label: per label, log P(positive answer) - log P(negative answer), each over its forms."""
        f = f.double()
        n = len(self.task.labels)
        if self.task.multi_label:
            group = torch.tensor(self.form_label, device=f.device) * 2 + torch.tensor(
                self.form_answer, device=f.device
            )
            lse = _group_logsumexp(f, group, 2 * n).view(f.shape[0], n, 2)
            return (lse[..., 0] - lse[..., 1]).tolist()
        idx = torch.tensor(self.form_label, device=f.device).expand_as(f)
        top = torch.full((f.shape[0], n), -torch.inf, dtype=f.dtype, device=f.device)
        top = top.scatter_reduce(1, idx, f, "amax")
        total = torch.zeros_like(top).scatter_add_(1, idx, (f - top.gather(1, idx)).exp())
        return (top + total.log()).tolist()

    def score(self, text: str) -> list[float]:
        """Every label's score: log of the summed probability of its surface forms given the prompt."""
        return self.score_many([text])[0]

    def score_many(self, texts: list[str]) -> list[list[float]]:
        return self._label_scores(self.form_scores_many(texts))

    def form_scores(self, text: str) -> torch.Tensor:
        """Log-probability of every continuation in ``self.forms`` given the prompt for ``text``."""
        return self.form_scores_many([text])[0]

    def form_scores_many(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError


NOT_SCORED = -1e9  # score of a continuation outside the shortlist: probability effectively 0, valid JSON


class _View(NamedTuple):
    """The label tree, or the part of it a text needs (its shortlisted labels' continuations)."""

    tokens: torch.Tensor  # node input tokens
    depth: torch.Tensor  # node depth (position offset after the text)
    anc: torch.Tensor  # node x node: ancestor-or-self
    rows: torch.Tensor  # per scored token: the node predicting it (-1: the text's last position)
    targets: torch.Tensor  # per scored token: its id
    owner: torch.Tensor  # per scored token: its continuation (index into forms)
    scored: torch.Tensor | None  # which continuations are scored (None: all)


class LabelScorer(_Prompt):
    def __init__(
        self,
        model,
        tokenizer,
        task: Task,
        use_chat_template: bool = True,
        surface_forms: bool = True,
        max_tokens: int = 2048,
        examples: list[tuple[str, int]] | None = None,
        shortlist=None,
        similar=None,
    ):
        """``similar``: a ``Retriever`` that picks each text's own few-shot examples (instead of
        ``examples``)."""
        self.model = model.eval()
        self.max_tokens = max_tokens  # tokens per forward pass when packing several texts
        self.device = next(model.parameters()).device
        self.shortlist = (
            shortlist  # callable: texts -> candidate label indices per text (or None: all labels)
        )
        if shortlist is not None and task.multi_label:
            raise ValueError("--shortlist is for single-label tasks")
        limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        self._prepare(tokenizer, task, use_chat_template, surface_forms, examples, limit, similar)
        self._build_tree()
        self._prefix_cache: Any = None  # the prompt prefix's KV cache (a transformers Cache)

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
        for li, lab in enumerate(self.forms):
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
        t = lambda x: torch.tensor(x, dtype=torch.long, device=dev)  # noqa: E731
        self._node_tokens, self._node_depth, self._node_anc = t(tokens), t(depth), anc.to(dev)
        self._rows, self._targets, self._owner = t(rows), t(targets), t(owner)
        self._full = _View(
            self._node_tokens, self._node_depth, self._node_anc, self._rows, self._targets, self._owner, None
        )
        # for shortlists: each continuation's scored tokens and the nodes they need
        self._entries: list[list[int]] = [[] for _ in self.forms]
        for e, f in enumerate(owner):
            self._entries[f].append(e)
        self._rows_list = rows

    def _view(self, labels: list[int] | None) -> _View:
        """The tree restricted to the continuations of ``labels`` (None: the whole tree)."""
        if labels is None:
            return self._full
        wanted = set(labels)
        forms = [f for f, lab in enumerate(self.form_label) if lab in wanted]
        entries = [e for f in forms for e in self._entries[f]]
        nodes = sorted({self._rows_list[e] for e in entries if self._rows_list[e] >= 0})  # parents come first
        pos = {node: i for i, node in enumerate(nodes)}
        dev = self.device
        sel = torch.tensor(nodes, dtype=torch.long, device=dev)
        scored = torch.zeros(len(self.forms), dtype=torch.bool, device=dev)
        scored[forms] = True
        rows = torch.tensor([pos.get(self._rows_list[e], -1) for e in entries], dtype=torch.long, device=dev)
        idx = torch.tensor(entries, dtype=torch.long, device=dev)
        return _View(
            self._node_tokens[sel],
            self._node_depth[sel],
            self._node_anc[sel][:, sel],
            rows,
            self._targets[idx],
            self._owner[idx],
            scored,
        )

    def fingerprint(self) -> str:
        """Identifies everything that changes the scores, so caches are never mixed up."""
        cfg = getattr(self.model, "config", None)
        key = {
            "model": getattr(cfg, "_name_or_path", ""),
            "dtype": str(next(self.model.parameters()).dtype),
            "quant": str(getattr(cfg, "quantization_config", "")),
            "prefix": self.prefix_ids,
            "suffix": self.suffix,
            "forms": self.forms,
            "form_label": self.form_label,
            "form_answer": self.form_answer,
        }
        if self.shortlist is not None:
            key["shortlist"] = self.shortlist.fingerprint()
        if self.similar is not None:
            key["similar"] = [self.similar.fingerprint(), self.middle]
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]

    @torch.no_grad()
    def form_scores_many(self, texts: list[str]) -> torch.Tensor:
        """(len(texts), len(self.forms)) log-probabilities, packing texts up to ``max_tokens`` per pass.
        With a shortlist, continuations of labels outside a text's shortlist get NOT_SCORED."""
        if self._prefix_cache is None:
            self._encode_prefix()
        ids = [self.text_ids(t, h) for t, h in zip(texts, self.heads(texts), strict=True)]
        labels = self.shortlist(texts) if self.shortlist is not None else [None] * len(texts)
        views = [self._view(lab) for lab in labels]
        if not self._packable:
            return torch.stack([self._score_alone(x, v) for x, v in zip(ids, views, strict=True)])
        out: list[torch.Tensor] = []
        pack: list[tuple[list[int], _View]] = []
        used = 0
        for x, view in zip(ids, views, strict=True):
            cost = len(x) + len(view.tokens)
            if pack and used + cost > self.max_tokens:
                out.append(self._score_pack(pack))
                pack, used = [], 0
            pack.append((x, view))
            used += cost
        if pack:
            out.append(self._score_pack(pack))
        return torch.cat(out) if out else torch.empty((0, len(self.forms)), device=self.device)

    @torch.no_grad()
    def _encode_prefix(self, chunk: int = 1024) -> None:
        """Run the prompt prefix once, in chunks against the growing cache, so that a long (few-shot) prompt
        never needs its whole attention matrix at once."""
        ids = torch.tensor([self.prefix_ids], device=self.device)
        cache = None
        for start in range(0, ids.shape[1], chunk):
            out = self.model(ids[:, start : start + chunk], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
        self._prefix_cache = cache
        self._prefix_last = torch.log_softmax(out.logits[0, -1].float(), -1)
        from transformers.cache_utils import DynamicLayer

        # Packing masks texts from each other and rolls the cache back to the prompt afterwards. Both need
        # plain attention layers: recurrent (linear attention) layers ignore the mask and cannot be rolled
        # back, and sliding-window layers drop old tokens. Other models score one text at a time.
        layers = getattr(cache, "layers", None) or []
        self._packable = bool(layers) and all(type(layer) is DynamicLayer for layer in layers)

    @torch.no_grad()
    def _score_alone(self, x: list[int], view: _View, rows_bytes: int = 2**31) -> torch.Tensor:
        """One text's continuation scores without packing: the text runs against a copy of the prompt's
        cache, then the multi-token continuations run in batches against copies of the text's cache (as
        many rows as fit in ``rows_bytes`` of cache)."""
        dev = self.device
        forms = range(len(self.forms)) if view.scored is None else view.scored.nonzero().flatten().tolist()
        total = torch.full((len(self.forms),), NOT_SCORED, dtype=torch.float32, device=dev)
        cache = copy.deepcopy(self._prefix_cache)
        first = self._prefix_last
        if x:
            out = self.model(torch.tensor([x], device=dev), past_key_values=cache, use_cache=True)
            first = torch.log_softmax(out.logits[0, -1].float(), -1)
        for f in forms:
            total[f] = first[self.forms[f][0]]
        longer = [f for f in forms if len(self.forms[f]) > 1]
        size = sum(t.numel() * t.element_size() for t in _tensors(cache))
        per = max(1, rows_bytes // max(size, 1))
        past = len(self.prefix_ids) + len(x)
        for i in range(0, len(longer), per):
            batch = longer[i : i + per]
            width = max(len(self.forms[f]) - 1 for f in batch)
            inp = torch.zeros((len(batch), width), dtype=torch.long, device=dev)
            mask = torch.zeros((len(batch), past + width), dtype=torch.long, device=dev)
            mask[:, :past] = 1
            for r, f in enumerate(batch):  # right-padded: the padding comes after every scored position
                n = len(self.forms[f]) - 1
                inp[r, :n] = torch.tensor(self.forms[f][:-1], device=dev)
                mask[r, past : past + n] = 1
            c = copy.deepcopy(cache)
            c.reorder_cache(torch.zeros(len(batch), dtype=torch.long, device=dev))  # one copy per row
            logits = self.model(inp, attention_mask=mask, past_key_values=c, use_cache=True).logits.float()
            for r, f in enumerate(batch):
                tgt = torch.tensor(self.forms[f][1:], device=dev)
                rows = logits[r, : len(tgt)]
                total[f] += (rows.gather(1, tgt[:, None])[:, 0] - torch.logsumexp(rows, -1)).sum()
        return total

    def _additive(self, allow: torch.Tensor) -> torch.Tensor:
        """The mask as additive biases (0 where attention is allowed): every attention implementation reads
        this form, while some (eager) would misread a boolean mask."""
        dtype = next(self.model.parameters()).dtype
        if not dtype.is_floating_point:  # quantised weights: use the compute dtype
            dtype = torch.bfloat16
        return torch.zeros(allow.shape, dtype=dtype, device=allow.device).masked_fill_(
            ~allow, torch.finfo(dtype).min
        )

    def _totals(self, per_token: torch.Tensor, view: _View) -> torch.Tensor:
        total = torch.zeros(len(self.forms), dtype=per_token.dtype, device=self.device)
        total.index_add_(0, view.owner, per_token)
        if view.scored is not None:
            total[~view.scored] = NOT_SCORED
        return total

    def _score_pack(self, pack: list[tuple[list[int], _View]]) -> torch.Tensor:
        dev, P = self.device, len(self.prefix_ids)
        L = sum(len(x) + len(v.tokens) for x, v in pack)
        tokens = torch.empty(L, dtype=torch.long, device=dev)
        positions = torch.empty(L, dtype=torch.long, device=dev)
        allow = torch.zeros((L, P + L), dtype=torch.bool, device=dev)
        allow[:, :P] = True  # everything sees the prompt
        keep: list[int] = []  # output rows to compute logits for
        first: list[int] = []  # per text: kept row predicting the first label token (-1: the prompt does)
        node_start: list[int] = []  # per text: kept row of its first tree node
        o = 0
        for x, view in pack:
            n, N = len(x), len(view.tokens)
            t = slice(o, o + n)  # this text's rows
            d = slice(o + n, o + n + N)  # its tree nodes
            tokens[t] = torch.tensor(x, dtype=torch.long, device=dev)
            tokens[d] = view.tokens
            positions[t] = torch.arange(P, P + n, device=dev)
            positions[d] = P + n + view.depth
            allow[t, P + o : P + o + n] = torch.ones((n, n), dtype=torch.bool, device=dev).tril()
            allow[d, P + o : P + o + n] = True
            allow[d, P + o + n : P + o + n + N] = view.anc
            first.append(len(keep) if n else -1)
            if n:
                keep.append(o + n - 1)
            node_start.append(len(keep))
            keep.extend(range(o + n, o + n + N))
            o += n + N
        if not keep:  # empty texts and single-token continuations: the prompt predicts everything
            return torch.stack([self._totals(self._prefix_last[v.targets], v) for _, v in pack])
        try:
            logits = self.model(
                tokens[None],
                attention_mask=self._additive(allow)[None, None],
                position_ids=positions[None],
                past_key_values=self._prefix_cache,
                use_cache=True,
                logits_to_keep=torch.tensor(keep, device=dev),
            ).logits[0]
        finally:
            extra = self._prefix_cache.get_seq_length() - P
            if extra:
                self._prefix_cache.crop(-extra)  # back to the prompt prefix
        # log p(target) = logit(target) - logsumexp(logits): only the targets' logits are needed, so the
        # vocabulary-sized softmax is never materialised for all rows at once (with many labels a pass keeps
        # thousands of rows, and full float32 log-softmaxes fragment GPU memory until it spills to system RAM)
        lse = torch.cat(
            [torch.logsumexp(logits[i : i + 256].float(), -1) for i in range(0, len(logits), 256)]
        )
        n = len(logits)  # row n stands for the prompt's last position
        results = []
        for (_, view), f, start in zip(pack, first, node_start, strict=True):
            rows = torch.where(view.rows < 0, f if f >= 0 else n, start + view.rows)
            at = rows.clamp(max=n - 1)
            lp = logits[at, view.targets].float() - lse[at]
            lp = torch.where(rows == n, self._prefix_last[view.targets], lp)
            results.append(self._totals(lp, view))
        return torch.stack(results)


def _tensors(obj) -> Iterator[torch.Tensor]:
    """The tensors a cache holds (in its layers' attributes, lists and dicts)."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            yield from _tensors(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from _tensors(o)
    elif hasattr(obj, "layers"):
        yield from _tensors(obj.layers)
    elif hasattr(obj, "__dict__"):
        yield from _tensors(vars(obj))


def combine_scores(per_teacher: list[list[float]], multi_label: bool = False) -> list[float]:
    """Ensemble of several teachers' label scores: the log of the mean of their label probabilities (each
    teacher's softmax; multi-label, each label's sigmoid, returned as log-odds). One teacher is unchanged."""
    if len(per_teacher) == 1:
        return per_teacher[0]
    scores = torch.tensor(per_teacher, dtype=torch.float64)
    if multi_label:
        return torch.logit(scores.sigmoid().mean(0)).tolist()
    return scores.softmax(-1).mean(0).log().tolist()


class VLLMScorer(_Prompt):
    """The same scoring through a vLLM server's OpenAI-compatible completions endpoint.

    For each text, every continuation is sent as a token-id prompt (prompt prefix + text + continuation) with
    ``prompt_logprobs``, which returns the log-probability of each prompt token; the continuation's tokens are
    summed. Start the server with prefix caching (``vllm serve MODEL --enable-prefix-caching``) so the shared
    prompt is computed once. Only the tokenizer is loaded locally."""

    def __init__(
        self,
        url: str,
        model: str,
        tokenizer,
        task: Task,
        examples=None,
        workers: int = 8,
        surface_forms: bool = True,
        timeout: float = 600,
        similar=None,
    ):
        self.url, self.model_name, self.workers, self.timeout = url.rstrip("/"), model, workers, timeout
        self.device = "vllm"
        self._prepare(tokenizer, task, True, surface_forms, examples, None, similar)

    def fingerprint(self) -> str:
        key = {
            "model": self.model_name,
            "server": "vllm",
            "prefix": self.prefix_ids,
            "suffix": self.suffix,
            "forms": self.forms,
            "form_label": self.form_label,
            "form_answer": self.form_answer,
        }
        if self.similar is not None:
            key["similar"] = [self.similar.fingerprint(), self.middle]
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]

    def _one(self, text: str, head: str = "") -> list[float]:
        import urllib.request

        prompt = self.prefix_ids + self.text_ids(text, head)
        body = {
            "model": self.model_name,
            "prompt": [prompt + f for f in self.forms],
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": 0,
        }
        req = urllib.request.Request(
            f"{self.url}/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            choices = sorted(json.loads(r.read())["choices"], key=lambda c: c["index"])
        out = []
        for form, choice in zip(self.forms, choices, strict=True):
            lps = choice["prompt_logprobs"][len(prompt) :]
            out.append(sum(entry[str(tok)]["logprob"] for tok, entry in zip(form, lps, strict=True)))
        return out

    def form_scores_many(self, texts: list[str]) -> torch.Tensor:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(self.workers) as pool:
            rows = list(pool.map(self._one, texts, self.heads(texts)))
        return torch.tensor(rows, dtype=torch.float64).reshape(len(texts), len(self.forms))


REASON_MARKER = "Answer:"  # a reasoning teacher writes its answer after this
REASON_INSTRUCTION = (
    f'\n\nThink it through briefly first. Then write "{REASON_MARKER}" followed by the answer.'
)


def reasoning_task(task: Task) -> Task:
    """The task with an instruction to reason before answering, in place of a trailing "Answer:"."""
    if task.multi_label:
        raise SystemExit("--reason is for single-label tasks")
    prompt = task.prompt.rstrip().removesuffix(REASON_MARKER).rstrip()
    return replace(task, prompt=prompt + REASON_INSTRUCTION)


def cut_reasoning(generated: str) -> tuple[str, str]:
    """(the reasoning, what the labels are scored after): the generation up to and including the first
    marker, or, if the model never wrote it, all of the generation with the marker appended."""
    at = generated.find(REASON_MARKER)
    if at >= 0:
        return generated[:at].strip(), generated[: at + len(REASON_MARKER)]
    reasoning = generated.strip()
    return reasoning, (generated.rstrip() + "\n" if reasoning else "") + REASON_MARKER


class _Reasoning(_Prompt):
    """A teacher that reasons before it answers: for each text it writes up to ``reason_tokens`` tokens
    greedily, cut by ``cut_reasoning``, and every label's continuations are then scored after prompt +
    reasoning + marker by the scorer's own code. ``reasoning`` holds the last texts' reasoning."""

    reason_tokens: int
    reasoning: dict[str, str]
    _after: dict[str, list[int]]  # text -> token ids of its reasoning and the marker

    def text_ids(self, text: str, head: str = "") -> list[int]:
        return super().text_ids(text, head) + self._after.get(text, [])

    def form_scores_many(self, texts: list[str]) -> torch.Tensor:
        self._after, self.reasoning = {}, {}
        prompts = [
            self.prefix_ids + self.text_ids(t, h) for t, h in zip(texts, self.heads(texts), strict=True)
        ]
        for text, out in zip(texts, self._generate(prompts) if texts else [], strict=True):
            self.reasoning[text], tail = cut_reasoning(out)
            self._after[text] = self.tok(tail, add_special_tokens=False)["input_ids"]
        return super().form_scores_many(texts)

    def _generate(self, prompts: list[list[int]]) -> list[str]:
        raise NotImplementedError

    def fingerprint(self) -> str:
        base = super().fingerprint()  # type: ignore[misc]  # the scorer after this mixin has it
        key = [base, "reason-v1", self.reason_tokens]
        return hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]


class ReasoningScorer(_Reasoning, LabelScorer):
    """``LabelScorer`` with reasoning first. The generation is batched with left padding but cannot reuse
    the cached prompt prefix, so this is much slower than scoring alone."""

    def __init__(self, model, tokenizer, task: Task, reason_tokens: int = 256, **kwargs):
        self.reason_tokens, self.reasoning, self._after = reason_tokens, {}, {}
        super().__init__(model, tokenizer, reasoning_task(task), **kwargs)

    @torch.no_grad()
    def _generate(self, prompts: list[list[int]]) -> list[str]:
        # ponytail: one generate call per score_many batch, the prompt is not shared across the batch;
        # reuse the prefix cache (replicated per row) if long few-shot prompts run out of memory
        tok, m = self.tok, max(map(len, prompts))
        pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
        ids = torch.tensor([[pad] * (m - len(p)) + p for p in prompts], device=self.device)
        mask = torch.tensor([[0] * (m - len(p)) + [1] * len(p) for p in prompts], device=self.device)
        stops = [t for t in dict.fromkeys([tok.eos_token_id, *end_of_turn(tok, self.chat)]) if t is not None]
        out = self.model.generate(
            ids,
            attention_mask=mask,
            max_new_tokens=self.reason_tokens,
            do_sample=False,
            eos_token_id=stops,
            pad_token_id=pad,
            stop_strings=[REASON_MARKER],
            tokenizer=tok,
        )
        return tok.batch_decode(out[:, m:], skip_special_tokens=True)


class VLLMReasoningScorer(_Reasoning, VLLMScorer):
    """``VLLMScorer`` with reasoning first: the server writes it (completions endpoint, greedy, stopping
    at the marker), then the labels are scored with ``prompt_logprobs`` as without reasoning."""

    def __init__(self, url: str, model: str, tokenizer, task: Task, reason_tokens: int = 256, **kwargs):
        self.reason_tokens, self.reasoning, self._after = reason_tokens, {}, {}
        super().__init__(url, model, tokenizer, reasoning_task(task), **kwargs)

    def _generate(self, prompts: list[list[int]]) -> list[str]:
        import urllib.request

        body = {
            "model": self.model_name,
            "prompt": prompts,
            "max_tokens": self.reason_tokens,
            "temperature": 0,
            "stop": [REASON_MARKER],
            "include_stop_str_in_output": True,
        }
        req = urllib.request.Request(
            f"{self.url}/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return [c["text"] for c in sorted(json.loads(r.read())["choices"], key=lambda c: c["index"])]


API_LABELS = 20  # hosted APIs return the top 20 log-probabilities of the first answer token
API_ATTEMPTS = 5  # tries per request on rate limits (HTTP 429), server errors (5xx) and timeouts
API_WAIT = 1.0  # seconds before the first retry; doubled after each
API_SAMPLE_TOKENS = 10  # reply length with --api-samples: enough to tell "A." from "A dog", cheap


class ApiScorer:
    """A hosted OpenAI-compatible chat API (OpenAI, DeepInfra, Together, a llama.cpp server, ...) as teacher.

    Such APIs return no prompt log-probabilities, only the top 20 log-probabilities of the first generated
    token, so the multi-token scoring above is impossible. Instead the labels are shown as lettered options
    (A, B, C, ...) and a label's score is the log-probability of its letter as the first answer token. A
    letter is one token in every tokenizer, so this is exact for single-label tasks with up to 20 labels and
    needs no local tokenizer. The API key is read from the environment variable ``api_key_env`` at each
    request and never stored, logged or fingerprinted; without it no Authorization header is sent.

    For APIs without log-probabilities, ``samples`` N (at least 2) asks for N answers at ``temperature``
    instead, and a label's score is the log of its share of the answers that begin with an option letter (see
    ``_letter``); a label no answer chose gets NOT_SCORED."""

    def __init__(
        self,
        url: str,
        model: str,
        task: Task,
        examples: list[tuple[str, int]] | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        workers: int = 8,
        timeout: float = 120,
        samples: int | None = None,
        temperature: float = 0.7,
        similar=None,
    ):
        if samples is not None and (samples < 2 or temperature <= 0):
            raise SystemExit("--api-samples needs at least 2 samples and --api-temperature above 0")
        if task.multi_label:
            raise SystemExit("--api-url works for single-label tasks; use a local or vLLM teacher")
        if len(task.labels) > API_LABELS:
            raise SystemExit(
                f"--api-url scores at most {API_LABELS} labels (the API returns {API_LABELS} "
                f"log-probabilities) and this task has {len(task.labels)}; use a local teacher "
                "(with --shortlist for very many labels) or --vllm-url"
            )
        self.url, self.model_name, self.task, self.examples = url.rstrip("/"), model, task, examples or []
        self.api_key_env, self.workers, self.timeout = api_key_env, workers, timeout
        self.samples, self.temperature = samples or 0, temperature  # samples 0: log-probability scoring
        self._n_works = True  # the server accepts "n" (several answers per request); False once it refused it
        self.similar = similar  # a Retriever: each text's own examples (instead of ``examples``)
        self.device = "api"
        self.letters = LETTERS[: len(task.labels)]
        self._blind: list[str] = []
        parts = urlsplit(self.url)
        self.host = f"{parts.hostname}:{parts.port}" if parts.port else str(parts.hostname)

    def fingerprint(self) -> str:
        key: dict[str, Any] = {
            "model": self.model_name,
            "api": self.host,
            "prompt": self.task.render(_SENTINEL, self.examples, lettered=True),
            "letters": self.letters,
            "labels": self.task.verbalisations,
            "scheme": "letters-v1",
        }
        if self.samples:
            key |= {"scheme": "letters-samples-v1", "samples": self.samples, "temperature": self.temperature}
        if self.similar is not None:
            key["similar"] = self.similar.fingerprint()
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]

    def _post(self, body: dict) -> dict:
        import time
        import urllib.error
        import urllib.request

        key = os.environ.get(self.api_key_env)
        headers = {"Content-Type": "application/json"} | ({"Authorization": f"Bearer {key}"} if key else {})
        req = urllib.request.Request(
            f"{self.url}/chat/completions", data=json.dumps(body).encode(), headers=headers
        )
        for attempt in range(API_ATTEMPTS):
            if attempt:
                time.sleep(API_WAIT * 2 ** (attempt - 1))
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise SystemExit(
                        f"the API at {self.host} refused the request (HTTP {e.code}): put the API key in the "
                        f"environment variable {self.api_key_env}, or name another one with --api-key-env"
                    ) from None
                if e.code != 429 and e.code < 500:
                    detail = e.read().decode("utf-8", "replace")[:500]
                    detail = detail.replace(key, "***") if key else detail
                    raise SystemExit(f"the API at {self.host} answered HTTP {e.code}: {detail}") from None
                last = f"HTTP {e.code}"
            except (urllib.error.URLError, TimeoutError) as e:
                last = str(getattr(e, "reason", e))
        raise SystemExit(f"the API at {self.host} failed {API_ATTEMPTS} times (last: {last})")

    def _letter(self, reply: str) -> str | None:
        """The option letter a sampled reply answers with, or None. The letter must stand alone at the start,
        after optional whitespace, "(", "[", "*" and "Answer:"/"The answer is", and be followed by ".", ")",
        ":", "]", "*" or the end of a line: "B", "B. negative", "(B)", "**B**" and "Answer: B" are B, but "A
        dog barked" is no letter, since A followed by a word is the English article. Case is ignored."""
        import re

        m = re.match(
            r"\s*(?:(?:the\s+)?answer(?:\s+is)?\s*:?\s*)?[(\[*]*([a-z])(?:[.):\]*]|[ \t]*(?:\n|$))",
            reply,
            re.I,
        )
        c = m.group(1).upper() if m else None
        return c if c and c in self.letters else None

    def _sampled(self, text: str, examples: list[tuple[str, int]] | None) -> list[float]:
        import math

        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": self._render(text, examples)}],
            "max_tokens": API_SAMPLE_TOKENS,
            "temperature": self.temperature,
        }

        def ask(extra: dict) -> list[str]:
            choices = self._post(body | extra).get("choices") or []
            if not choices:
                raise SystemExit(f"the API at {self.host} returned no answer for {self.model_name}")
            return [(c.get("message") or {}).get("content") or "" for c in choices]

        replies: list[str] = []
        if self._n_works:
            try:
                replies = ask({"n": self.samples})
            except SystemExit as e:  # the server refused "n" (or the request: then the next one fails too)
                if "answered HTTP" not in str(e):
                    raise
                self._n_works = False
        while len(replies) < self.samples:  # "n" refused or fewer answers than asked: one request per answer
            replies += ask({})
        letters = [c for c in map(self._letter, replies[: self.samples]) if c]
        if not letters:
            self._blind.append(text)
            return [-math.log(len(self.letters))] * len(self.letters)
        return [
            math.log(letters.count(c) / len(letters)) if c in letters else NOT_SCORED for c in self.letters
        ]

    def _render(self, text: str, examples: list[tuple[str, int]] | None) -> str:
        return self.task.render(text, self.examples if examples is None else examples, lettered=True)

    def _one(self, text: str, examples: list[tuple[str, int]] | None = None) -> list[float]:
        if self.samples:
            return self._sampled(text, examples)
        content = self._render(text, examples)
        out = self._post(
            {
                "model": self.model_name,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": API_LABELS,
            }
        )
        try:
            top = out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            lps = [(t["token"].strip().rstrip(".):").upper(), float(t["logprob"])) for t in top]
        except (KeyError, IndexError, TypeError):
            lps = []
        if not lps:
            raise SystemExit(
                f"the API at {self.host} returned no top log-probabilities for {self.model_name}; "
                "the model must support logprobs and top_logprobs"
            )
        best: dict[str, float] = {}
        for tok, lp in lps:  # "A", " A" and "a" all count for option A
            best[tok] = max(best.get(tok, -float("inf")), lp)
        # A letter outside the returned top tokens is at most as likely as the least likely returned one:
        # it gets that log-probability, an upper bound.
        floor = min(lp for _, lp in lps)
        if not any(c in best for c in self.letters):
            self._blind.append(text)  # list.append is thread-safe; score_many reports these
        return [best.get(c, floor) for c in self.letters]

    def score_many(self, texts: list[str]) -> list[list[float]]:
        import warnings
        from concurrent.futures import ThreadPoolExecutor

        self._blind = []
        shots = self.similar(texts) if self.similar is not None else [None] * len(texts)
        with ThreadPoolExecutor(self.workers) as pool:
            rows = list(pool.map(self._one, texts, shots))
        blind = len(self._blind)  # texts with no option letter among the returned tokens or answers
        if blind:
            what = (
                f"none of {self.model_name}'s {self.samples} sampled answers began with an option letter"
                if self.samples
                else f"{self.model_name}'s first answer token was not an option letter"
            )
            warnings.warn(
                f"{blind} of {len(texts)} texts: {what}, so every label got the same score; make the prompt "
                "ask for the letter only (see the docs on hosted APIs)",
                stacklevel=2,
            )
        return rows

    def score(self, text: str) -> list[float]:
        return self.score_many([text])[0]


class Ensemble:
    """Several teachers used as one: their label probabilities are averaged (as in labelling)."""

    def __init__(self, members: list[Scorer]):
        self.members = members

    def score_many(self, texts: list[str]) -> list[list[float]]:
        per = [m.score_many(texts) for m in self.members]
        multi = self.members[0].task.multi_label
        return [combine_scores([p[i] for p in per], multi) for i in range(len(texts))]

    def score(self, text: str) -> list[float]:
        return self.score_many([text])[0]


def load_teacher(
    name: str,
    task: Task,
    device: str = "auto",
    load_4bit: bool = False,
    max_tokens: int = 2048,
    examples: list[tuple[str, int]] | None = None,
    vllm_url: str | None = None,
    shortlist: int | None = None,
    shortlist_model: str | None = None,
    shortlist_keep: list[str] | None = None,
    api_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    api_samples: int | None = None,
    api_temperature: float = 0.7,
    similar: int = 0,
    similar_model: str | None = None,
    reason_tokens: int = 0,
) -> LabelScorer | VLLMScorer | ApiScorer:
    """A local teacher, or with ``vllm_url`` the same model served by vLLM, or with ``api_url`` the model
    ``name`` at a hosted OpenAI-compatible chat API (key in the environment variable ``api_key_env``); then
    ``device`` is not used. ``shortlist`` K: score only the K labels most similar to each text (for tasks with
    very many labels). ``api_samples`` N: the API teacher samples N answers at ``api_temperature`` instead of
    reading log-probabilities. ``similar`` M: ``examples`` is a pool of gold examples, and each text's prompt
    shows the M most similar to it (embedded with ``similar_model``). ``reason_tokens`` K: the teacher
    reasons in up to K tokens before it answers (``ReasoningScorer``, ``VLLMReasoningScorer``)."""
    if api_samples and not api_url:
        raise SystemExit("--api-samples works with --api-url")
    retriever = None
    if similar:
        from .shortlist import DEFAULT_SHORTLIST_MODEL, Retriever

        where = "cpu" if api_url or vllm_url else pick_device(device)
        retriever = Retriever(examples or [], similar, similar_model or DEFAULT_SHORTLIST_MODEL, where)
        examples = None
    if api_url:
        if vllm_url:
            raise SystemExit("use either --vllm-url or --api-url, not both")
        if shortlist:
            raise SystemExit("--shortlist works with a local teacher, not --api-url")
        if reason_tokens:
            raise SystemExit(
                "--reason needs a local or vLLM teacher: a hosted API returns no log-probabilities "
                "of the answer after the reasoning"
            )
        return ApiScorer(
            api_url,
            name,
            task,
            examples,
            api_key_env,
            samples=api_samples,
            temperature=api_temperature,
            similar=retriever,
        )
    tok = AutoTokenizer.from_pretrained(name)
    if vllm_url:
        if shortlist:
            raise SystemExit("--shortlist works with a local teacher, not --vllm-url")
        if reason_tokens:
            return VLLMReasoningScorer(
                vllm_url, name, tok, task, reason_tokens, examples=examples, similar=retriever
            )
        return VLLMScorer(vllm_url, name, tok, task, examples, similar=retriever)
    device = pick_device(device)
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
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if not load_4bit:
        model.to(device)
    short = None
    if shortlist:
        from .shortlist import DEFAULT_SHORTLIST_MODEL, Shortlist

        short = Shortlist(task, shortlist, shortlist_model or DEFAULT_SHORTLIST_MODEL, device, shortlist_keep)
    if reason_tokens:
        return ReasoningScorer(
            model,
            tok,
            task,
            reason_tokens,
            max_tokens=max_tokens,
            examples=examples,
            shortlist=short,
            similar=retriever,
        )
    return LabelScorer(
        model, tok, task, max_tokens=max_tokens, examples=examples, shortlist=short, similar=retriever
    )


def prompt_members(scorer: Scorer, variants: list[Task]) -> list[Scorer]:
    """A prompt ensemble on one loaded model: ``scorer`` (the task's own prompt), then a scorer for each
    variant task file (same labels, another prompt), sharing the model, few-shot examples (or the retriever
    of similar ones), shortlist and reasoning setting."""
    if not variants:
        return [scorer]
    if not isinstance(scorer, LabelScorer):
        raise SystemExit("prompt ensembles (--prompts) need a local teacher, not --vllm-url or --api-url")
    make: Any = type(scorer)  # a ReasoningScorer's members reason too
    extra = {"reason_tokens": scorer.reason_tokens} if isinstance(scorer, ReasoningScorer) else {}
    return [
        scorer,
        *(
            make(
                scorer.model,
                scorer.tok,
                t,
                max_tokens=scorer.max_tokens,
                examples=scorer.examples,
                shortlist=scorer.shortlist,
                similar=scorer.similar,
                **extra,
            )
            for t in variants
        ),
    ]


def quick_key(
    name: str,
    task: Task,
    device: str,
    load_4bit: bool,
    examples: list[tuple[str, int]] | None = None,
    shortlist: int | None = None,
    shortlist_model: str | None = None,
    shortlist_keep: list[str] | None = None,
    similar: int = 0,
    similar_model: str | None = None,
    reason_tokens: int = 0,
) -> str:
    """What decides a local teacher's cache fingerprint, from the tokenizer alone (no model weights): the
    rendered prompt and continuations, and how the model would be loaded. `label` maps it to the full
    fingerprint, so a run whose texts are all cached never loads the teacher."""
    p = _Prompt()
    tok, prompted = AutoTokenizer.from_pretrained(name), reasoning_task(task) if reason_tokens else task
    p._prepare(tok, prompted, True, True, None if similar else examples, None)
    key = {
        "model": name,
        "device": pick_device(device),
        "load_4bit": load_4bit,
        "prefix": p.prefix_ids,
        "suffix": p.suffix,
        "forms": p.forms,
        "form_label": p.form_label,
        "form_answer": p.form_answer,
        "shortlist": ([shortlist, shortlist_model] + ([sorted(shortlist_keep)] if shortlist_keep else []))
        if shortlist
        else None,
    }
    if similar:
        from .shortlist import DEFAULT_SHORTLIST_MODEL, retrieval_key

        key["similar"] = retrieval_key(examples or [], similar, similar_model or DEFAULT_SHORTLIST_MODEL)
    if reason_tokens:
        key["reason"] = ["reason-v1", reason_tokens]
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


class ScoreCache:
    """Append-only JSONL of {"h": text hash, "s": label scores} (and "r": the reasoning, for a teacher that
    reasons); reopening it resumes where it stopped."""

    def __init__(self, path: Path):
        self.path = path
        self.scores: dict[str, list[float]] = {}
        self.reasoning: dict[str, str] = {}
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:  # a line cut off by an interrupted run
                        continue
                    self.scores[row["h"]] = row["s"]
                    if "r" in row:
                        self.reasoning[row["h"]] = row["r"]

    def fill(self, scorer: Scorer, texts: Iterable[str], progress=None, batch: int = 64, on_read=None) -> int:
        """Score every text not in the cache yet, ``batch`` texts at a time (the scorer packs them into
        forward passes). Returns how many were scored now."""
        n = 0
        pending: list[str] = []
        with self.path.open("a", encoding="utf-8") as f:

            def flush() -> None:
                nonlocal n
                scores = scorer.score_many(pending)
                why = getattr(scorer, "reasoning", {})  # a reasoning teacher's text before its answer
                for text, s in zip(pending, scores, strict=True):
                    h = text_hash(text)
                    self.scores[h] = s
                    row: dict[str, Any] = {"h": h, "s": s}
                    if text in why:
                        row["r"] = self.reasoning[h] = why[text]
                    f.write(json.dumps(row) + "\n")
                    n += 1
                    if progress:
                        progress(n)
                f.flush()
                pending.clear()

            for text in texts:
                if on_read:
                    on_read()  # every text read, cached or not (for progress and time left)
                if text_hash(text) not in self.scores and text not in pending:
                    pending.append(text)
                    if len(pending) == batch:
                        flush()
            if pending:
                flush()
        return n

    def get(self, text: str) -> list[float]:
        return self.scores[text_hash(text)]


def iter_unique(texts: Iterable[str]) -> Iterator[str]:
    seen: set[str] = set()
    for t in texts:
        if t not in seen:
            seen.add(t)
            yield t
