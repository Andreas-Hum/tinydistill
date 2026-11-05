"""Candidate labels per text, for tasks with very many labels: a small sentence-embedding model compares each
text with the label names (and descriptions), and the teacher scores only the K most similar labels. The same
model picks each text's most similar gold examples for the teacher's prompt (``Retriever``)."""

from __future__ import annotations

import hashlib
import json

import torch
import torch.nn.functional as F

from .task import Task

DEFAULT_SHORTLIST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """A small sentence-embedding model."""

    def __init__(self, model: str = DEFAULT_SHORTLIST_MODEL, device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer

        self.model_name, self.device = model, device
        self.tok = AutoTokenizer.from_pretrained(model)
        self.encoder = AutoModel.from_pretrained(model).to(device).eval()

    @torch.no_grad()
    def embed(self, texts: list[str], batch: int = 64) -> torch.Tensor:
        """Mean-pooled, length-normalised sentence embeddings."""
        out = []
        for b in range(0, len(texts), batch):
            enc = self.tok(
                texts[b : b + batch], padding=True, truncation=True, max_length=128, return_tensors="pt"
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            h = self.encoder(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            out.append(F.normalize((h * m).sum(1) / m.sum(1).clamp_min(1e-9), dim=-1))
        return torch.cat(out)


class Shortlist(Embedder):
    def __init__(
        self,
        task: Task,
        k: int,
        model: str = DEFAULT_SHORTLIST_MODEL,
        device: str = "cpu",
        keep: list[str] | None = None,
    ):
        """``keep``: label ids scored for every text besides its K most similar labels (catch-all labels such
        as "other" or "out of scope", whose names resemble no particular text)."""
        unknown = sorted(set(keep or []) - set(task.labels))
        if unknown:
            raise SystemExit(
                f"--shortlist-keep: unknown label ids {unknown}; the task's ids are its `labels`"
            )
        self.keep = sorted(task.labels.index(x) for x in keep or [])
        self.k = k
        super().__init__(model, device)
        descs = task.descriptions or [None] * len(task.labels)
        self.labels = self.embed(
            [f"{v}: {d}" if d else v for v, d in zip(task.verbalisations, descs, strict=True)]
        )

    def __call__(self, texts: list[str]) -> list[list[int]]:
        """The indices of the K labels most similar to each text."""
        sims = self.embed(texts) @ self.labels.T
        k = min(self.k, sims.shape[1])
        return [sorted(set(row.tolist()) | set(self.keep)) for row in sims.topk(k, dim=1).indices]

    def fingerprint(self) -> dict:
        key = {"k": self.k, "model": self.model_name}
        return key | {"keep": self.keep} if self.keep else key


def retrieval_key(pool: list[tuple[str, int]], m: int, model: str, sets: int = 1, bucket: int = 0) -> dict:
    """What decides the retrieved examples: the gold pool, M and the embedder (part of the fingerprint), and
    with several example sets, which of them."""
    pool_hash = hashlib.sha256(json.dumps(pool).encode()).hexdigest()[:16]
    key = {"m": m, "model": model, "pool": pool_hash}
    return key | {"bucket": [bucket, sets]} if sets > 1 else key


class Retriever(Embedder):
    """Per text, the M gold examples most similar to it (cosine similarity of sentence embeddings), as
    (text, label) pairs ordered from least to most similar, so the most similar ends up next to the text
    (retrieved demonstrations, Liu et al. 2021). A text is never its own example.

    With ``sets`` K above 1 (demonstration ensembling, Khalifa et al. 2023) it picks bucket ``bucket`` k of
    each text's K*M most similar examples: ranks k*M+1 to (k+1)*M, still ordered most similar last."""

    def __init__(
        self,
        pool: list[tuple[str, int]],
        m: int,
        model: str = DEFAULT_SHORTLIST_MODEL,
        device: str = "cpu",
        sets: int = 1,
    ):
        super().__init__(model, device)
        self.pool, self.m, self.sets, self.bucket = list(pool), m, sets, 0
        self.vectors = self.embed([t for t, _ in self.pool]) if self.pool else None
        self.where: dict[str, list[int]] = {}  # text -> its positions in the pool
        for j, (t, _) in enumerate(self.pool):
            self.where.setdefault(t, []).append(j)

    def __call__(self, texts: list[str]) -> list[list[tuple[str, int]]]:
        if self.vectors is None or not texts:
            return [[] for _ in texts]
        sims = self.embed(texts) @ self.vectors.T
        for i, t in enumerate(texts):
            sims[i, self.where.get(t, [])] = -torch.inf  # exact matches: never a text's own example
        top = sims.topk(min(self.m * self.sets, len(self.pool)), dim=1)
        at = slice(self.bucket * self.m, (self.bucket + 1) * self.m)  # ranks, most similar first
        ranked = (
            [self.pool[j] for s, j in zip(vals.tolist(), idx.tolist(), strict=True) if s > -torch.inf]
            for vals, idx in zip(top.values, top.indices, strict=True)
        )
        return [r[at][::-1] for r in ranked]

    def fingerprint(self) -> dict:
        return retrieval_key(self.pool, self.m, self.model_name, self.sets, self.bucket)
