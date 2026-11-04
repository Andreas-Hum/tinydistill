"""Synthetic unlabelled texts: the teacher writes example inputs for each label from the task's instruction.

This is the one place where the teacher generates free text, and only to produce inputs. The label of a
synthetic text is not taken from the request that produced it: the output has no gold-label field, and the
texts are scored by the teacher like any other unlabelled data.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from pathlib import Path

import torch
import yaml

from .task import Task

REQUEST = """Below is the instruction of a text classification task. Write {n} different, realistic texts of
the kind it classifies, each of which belongs to the category "{name}"{description}. Vary the wording, length
and details. Write one text per line, with no numbering, bullets or quotes.{extra}

Instruction:
{instruction}"""

# For --vary: one value of each is sampled per request (AttrPrompt, Yu et al. 2023). A task file's
# `attributes:` mapping replaces them.
ATTRIBUTES = {
    "length": ["short", "medium", "long"],
    "tone": ["casual", "formal", "angry", "polite", "terse"],
    "writer's situation": [
        "in a hurry",
        "writing for the first time",
        "following up on an earlier message",
        "unsure of the right words",
        "writing on a phone",
    ],
}


def load_attributes(path: str | Path) -> dict[str, list[str]]:
    """The task file's `attributes:` mapping {name: [values]}, or the built-in ``ATTRIBUTES``."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")).get("attributes")
    if raw is None:
        return ATTRIBUTES
    if not (
        isinstance(raw, dict)
        and raw
        and all(isinstance(v, list) and v and all(isinstance(x, str) for x in v) for v in raw.values())
    ):
        raise SystemExit(f"{path}: 'attributes' must map each attribute's name to a list of values")
    return {str(k): v for k, v in raw.items()}


def request(task: Task, label: int, n: int, attributes: dict[str, str] | None = None, extra: str = "") -> str:
    """The request for ``n`` texts of a label, optionally with ``attributes`` (name: value) every text should
    have and an ``extra`` instruction (a sentence or more, appended to the request)."""
    desc = task.descriptions[label] if task.descriptions else None
    if attributes:
        given = "; ".join(f"{k}: {v}" for k, v in attributes.items())
        extra = f" Give every text these attributes: {given}." + extra
    return REQUEST.format(
        n=n,
        name=task.verbalisations[label],
        description=f" ({desc})" if desc else "",
        instruction=task.render("[the text]"),
        extra=extra,
    )


def parse_lines(output: str, task: Task) -> list[str]:
    """One text per line; numbering, bullets and quotes removed; label names and very short lines dropped."""
    names = {v.lower() for v in task.verbalisations}
    texts = []
    for line in output.splitlines():
        t = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"“”').strip()
        if len(t) >= 3 and t.lower() not in names:
            texts.append(t)
    return texts


def generator(model, tok, temperature: float = 0.9, max_new_tokens: int = 400) -> Callable[[str], str]:
    """Sample one reply of the model to a user message."""
    chat = bool(getattr(tok, "chat_template", None))

    @torch.no_grad()
    def generate(message: str) -> str:
        if chat:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": message}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # write the texts directly (Qwen3 style templates; others ignore it)
            )
            ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        else:
            ids = tok(message + "\n", return_tensors="pt").input_ids
        ids = ids.to(next(model.parameters()).device)
        out = model.generate(
            ids,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)

    return generate


class NearDuplicates:
    """Keeps a text only if its cosine similarity to every text kept so far, and to every ``existing`` text,
    is below ``threshold``. ``embed`` returns length-normalised embeddings (``shortlist.Embedder.embed``)."""

    def __init__(
        self, embed: Callable[[list[str]], torch.Tensor], threshold: float, existing: list[str] | None = None
    ):
        self.embed, self.threshold, self.dropped = embed, threshold, 0
        self.kept = embed(existing) if existing else None

    def __call__(self, text: str) -> bool:
        # one embedding per call and a linear scan over the kept texts: batch them if it gets slow
        v = self.embed([text])
        if self.kept is not None and (self.kept @ v[0]).max() >= self.threshold:
            self.dropped += 1
            return False
        self.kept = v if self.kept is None else torch.cat([self.kept, v])
        return True


def synthesise(
    task: Task,
    generate: Callable[[str], str],
    per_label: int,
    out: Path,
    batch: int = 10,
    max_rounds: int | None = None,
    log: Callable[[str], None] = print,
    attributes: dict[str, list[str]] | None = None,
    seed: int = 0,
    keep: Callable[[str], bool] | None = None,
    jobs: list[tuple[int, str]] | None = None,
) -> int:
    """Ask for ``batch`` texts per request until each label has ``per_label`` new distinct texts, or until
    ``max_rounds`` requests were made for it (by default three times as many as would be needed). Writes JSON
    lines with the task's text field and a ``prompted_label`` note. Returns how many texts were written.

    ``attributes``: each request names one value of each, sampled with ``seed`` (also written to the row).
    ``keep``: a text is written only if it returns True (e.g. ``NearDuplicates``). ``jobs``: (label index,
    extra instruction) pairs to ask for instead of every label once with none."""
    seen: set[str] = set()
    written = 0
    rng = random.Random(seed)
    rounds = max_rounds or 3 * -(-per_label // batch)
    with out.open("w", encoding="utf-8") as f:
        for li, extra in jobs or [(li, "") for li in range(len(task.labels))]:
            name = task.labels[li]
            got = 0
            for _ in range(rounds):
                if got >= per_label:
                    break
                attrs = {k: rng.choice(v) for k, v in attributes.items()} if attributes else None
                reply = generate(request(task, li, min(batch, per_label - got), attrs, extra))
                for t in parse_lines(reply, task):
                    if got >= per_label or t.lower() in seen or (keep and not keep(t)):
                        continue
                    seen.add(t.lower())
                    row = {task.text_field: t, "prompted_label": name} | (
                        {"attributes": attrs} if attrs else {}
                    )
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    got += 1
            written += got
            log(f"{name}: {got} texts" + ("" if got >= per_label else f" (wanted {per_label})"))
    return written
