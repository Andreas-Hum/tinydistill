"""Which texts to label next: where the student and the teacher disagree, then where the teacher is least
sure, spread over the labels the teacher predicts."""

from __future__ import annotations

import numpy as np


def priority(p: np.ndarray, q: np.ndarray, multi_label: bool = False) -> np.ndarray:
    """A score per text, higher = more useful to label. ``p`` are the teacher's probabilities, ``q`` the
    student's (texts x labels). Single-label: 1 if their top labels differ, plus how close the teacher's two
    best labels are. Multi-label: the share of labels on which they disagree, plus how close the teacher's
    probabilities are to 0.5."""
    if multi_label:
        disagree = ((p >= 0.5) != (q >= 0.5)).mean(-1)
        unsure = 1 - 2 * np.abs(p - 0.5).mean(-1)
        return disagree + unsure / 2
    top2 = np.sort(p, -1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    return (p.argmax(-1) != q.argmax(-1)).astype(float) + (1 - margin) / 2


def pick(score: np.ndarray, group: np.ndarray, n: int) -> list[int]:
    """The n best texts by score, taking turns between groups (the teacher's predicted label) so that a
    batch covers many labels instead of one confusing pair."""
    queues = {g: sorted(np.flatnonzero(group == g), key=lambda i: -score[i]) for g in np.unique(group)}
    order = sorted(queues, key=lambda g: -score[queues[g][0]])  # groups with the most useful text first
    out: list[int] = []
    while len(out) < n and any(queues.values()):
        for g in order:
            if queues[g] and len(out) < n:
                out.append(int(queues[g].pop(0)))
    return out


def uncertainty(logits, multi_label: bool = False):
    """How unsure a model is of each text (a torch tensor, texts x labels, of logits): the entropy of its
    predicted distribution; multi-label, the mean binary entropy of its labels."""
    import torch

    if multi_label:
        return torch.distributions.Bernoulli(logits=logits).entropy().mean(-1)
    return torch.distributions.Categorical(logits=logits).entropy()
