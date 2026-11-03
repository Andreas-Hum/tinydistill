"""Teacher: score each label's verbalisation under a causal LM, cache the scores on disk.

For a text x and a continuation c_1..c_m (a way of writing a label, followed by the end-of-turn token),
log p(c | prompt(x)) = sum_j log p(c_j | prompt(x), c_<j). A label can be written in several surface forms
(lowercase, Capitalised, UPPERCASE, as written; each with and without a leading space), and its score is the
log of the summed probability of its distinct forms. A softmax over the label scores gives the teacher's soft
targets. There is no free-form generation.

Speed: the prompt is split around the text into a constant prefix and a per-text suffix. The prefix's KV
cache is computed once. A text and all its continuations form one segment: the text's tokens, then a prefix
tree of the continuations' tokens (continuations that start alike share nodes). Several segments are packed
into one sequence, up to a token budget, and a 4D attention mask lets every position attend to the cached
prompt plus, within its own segment, the text (causally) and its own tree ancestors. One forward pass scores
the whole pack; there is no padding and no copy of the cache, which is cropped back to the prompt afterwards.
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
_LABEL = "\x00LABEL\x00"


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
    local scorer and the vLLM client, which score the same continuations."""

    def _prepare(
        self,
        tokenizer,
        task: Task,
        use_chat_template: bool,
        surface_forms: bool,
        examples: list[tuple[str, int]] | None,
        max_positions: int | None,
    ) -> None:
        self.tok = tokenizer
        self.task = task
        self.chat = use_chat_template and bool(getattr(tokenizer, "chat_template", None))

        # Split the rendered prompt around the text (and, multi-label, around the label being asked about).
        self.examples = examples or []
        user = task.render(_SENTINEL, self.examples, _LABEL if task.multi_label else None)
        if self.chat:
            full = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
            )
        else:
            full = user
        prefix, suffix = full.split(_SENTINEL)
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

    def text_ids(self, text: str) -> list[int]:
        return self.tok(text + self.suffix, add_special_tokens=False)["input_ids"]

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
    ):
        self.model = model.eval()
        self.max_tokens = max_tokens  # tokens per forward pass when packing several texts
        self.device = next(model.parameters()).device
        limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        self._prepare(tokenizer, task, use_chat_template, surface_forms, examples, limit)
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
        self._node_tokens = torch.tensor(tokens, dtype=torch.long, device=dev)
        self._node_depth = torch.tensor(depth, dtype=torch.long, device=dev)
        self._node_anc = anc.to(dev)
        self._rows = torch.tensor(rows, dtype=torch.long, device=dev)
        self._targets = torch.tensor(targets, dtype=torch.long, device=dev)
        self._owner = torch.tensor(owner, dtype=torch.long, device=dev)

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
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]

    @torch.no_grad()
    def form_scores_many(self, texts: list[str]) -> torch.Tensor:
        """(len(texts), len(self.forms)) log-probabilities, packing texts up to ``max_tokens`` per pass."""
        if self._prefix_cache is None:
            self._encode_prefix()
        ids = [self.text_ids(t) for t in texts]
        N = len(self._node_tokens)
        out, pack, used = [], [], 0
        for x in ids:
            if pack and used + len(x) + N > self.max_tokens:
                out.append(self._score_pack(pack))
                pack, used = [], 0
            pack.append(x)
            used += len(x) + N
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

    def _additive(self, allow: torch.Tensor) -> torch.Tensor:
        """The mask as additive biases (0 where attention is allowed): every attention implementation reads
        this form, while some (eager) would misread a boolean mask."""
        dtype = next(self.model.parameters()).dtype
        if not dtype.is_floating_point:  # quantised weights: use the compute dtype
            dtype = torch.bfloat16
        return torch.zeros(allow.shape, dtype=dtype, device=allow.device).masked_fill_(
            ~allow, torch.finfo(dtype).min
        )

    def _score_pack(self, pack: list[list[int]]) -> torch.Tensor:
        dev, P, N = self.device, len(self.prefix_ids), len(self._node_tokens)
        L = sum(len(x) for x in pack) + N * len(pack)
        tokens = torch.empty(L, dtype=torch.long, device=dev)
        positions = torch.empty(L, dtype=torch.long, device=dev)
        allow = torch.zeros((L, P + L), dtype=torch.bool, device=dev)
        allow[:, :P] = True  # everything sees the prompt
        keep: list[int] = []  # output rows to compute logits for
        first: list[int] = []  # per text: kept row predicting the first label token (-1: the prompt does)
        node_start: list[int] = []  # per text: kept row of its first tree node
        o = 0
        for x in pack:
            n = len(x)
            t = slice(o, o + n)  # this text's rows
            d = slice(o + n, o + n + N)  # its tree nodes
            tokens[t] = torch.tensor(x, dtype=torch.long, device=dev)
            tokens[d] = self._node_tokens
            positions[t] = torch.arange(P, P + n, device=dev)
            positions[d] = P + n + self._node_depth
            allow[t, P + o : P + o + n] = torch.ones((n, n), dtype=torch.bool, device=dev).tril()
            allow[d, P + o : P + o + n] = True
            allow[d, P + o + n : P + o + n + N] = self._node_anc
            first.append(len(keep) if n else -1)
            if n:
                keep.append(o + n - 1)
            node_start.append(len(keep))
            keep.extend(range(o + n, o + n + N))
            o += n + N
        if not keep:  # empty texts and single-token continuations: the prompt predicts everything
            per_token = self._prefix_last[self._targets]
            one = torch.zeros(len(self.forms), dtype=per_token.dtype, device=dev).index_add_(
                0, self._owner, per_token
            )
            return one.expand(len(pack), -1)
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
        lp = torch.cat([torch.log_softmax(logits.float(), -1), self._prefix_last[None]])  # last row: prompt
        results = []
        for f, start in zip(first, node_start, strict=True):
            rows = torch.where(self._rows < 0, f if f >= 0 else len(lp) - 1, start + self._rows)
            per_token = lp[rows, self._targets]
            total = torch.zeros(len(self.forms), dtype=per_token.dtype, device=dev)
            results.append(total.index_add_(0, self._owner, per_token))
        return torch.stack(results)


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
    ):
        self.url, self.model_name, self.workers, self.timeout = url.rstrip("/"), model, workers, timeout
        self.device = "vllm"
        self._prepare(tokenizer, task, True, surface_forms, examples, None)

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
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]

    def _one(self, text: str) -> list[float]:
        import urllib.request

        prompt = self.prefix_ids + self.text_ids(text)
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
            rows = list(pool.map(self._one, texts))
        return torch.tensor(rows, dtype=torch.float64).reshape(len(texts), len(self.forms))


class Ensemble:
    """Several teachers used as one: their label probabilities are averaged (as in labelling)."""

    def __init__(self, members: list[LabelScorer]):
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
) -> LabelScorer | VLLMScorer:
    """A local teacher, or with ``vllm_url`` the same model served by vLLM (then ``device`` is not used)."""
    tok = AutoTokenizer.from_pretrained(name)
    if vllm_url:
        return VLLMScorer(vllm_url, name, tok, task, examples)
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
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if not load_4bit:
        model.to(device)
    return LabelScorer(model, tok, task, max_tokens=max_tokens, examples=examples)


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

    def fill(
        self, scorer: LabelScorer, texts: Iterable[str], progress=None, batch: int = 64, on_read=None
    ) -> int:
        """Score every text not in the cache yet, ``batch`` texts at a time (the scorer packs them into
        forward passes). Returns how many were scored now."""
        n = 0
        pending: list[str] = []
        with self.path.open("a", encoding="utf-8") as f:

            def flush() -> None:
                nonlocal n
                for text, s in zip(pending, scorer.score_many(pending), strict=True):
                    h = text_hash(text)
                    self.scores[h] = s
                    f.write(json.dumps({"h": h, "s": s}) + "\n")
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
