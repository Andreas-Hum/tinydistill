"""Candidate labels per text, for tasks with very many labels: a small sentence-embedding model compares each
text with the label names (and descriptions), and the teacher scores only the K most similar labels."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .task import Task

DEFAULT_SHORTLIST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Shortlist:
    def __init__(self, task: Task, k: int, model: str = DEFAULT_SHORTLIST_MODEL, device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer

        self.k, self.model_name, self.device = k, model, device
        self.tok = AutoTokenizer.from_pretrained(model)
        self.encoder = AutoModel.from_pretrained(model).to(device).eval()
        descs = task.descriptions or [None] * len(task.labels)
        self.labels = self.embed(
            [f"{v}: {d}" if d else v for v, d in zip(task.verbalisations, descs, strict=True)]
        )

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

    def __call__(self, texts: list[str]) -> list[list[int]]:
        """The indices of the K labels most similar to each text."""
        sims = self.embed(texts) @ self.labels.T
        k = min(self.k, sims.shape[1])
        return [sorted(row.tolist()) for row in sims.topk(k, dim=1).indices]

    def fingerprint(self) -> dict:
        return {"k": self.k, "model": self.model_name}
