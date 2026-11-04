"""Synthetic unlabelled texts: the teacher writes example inputs for each label from the task's instruction.

This is the one place where the teacher generates free text, and only to produce inputs. The label of a
synthetic text is not taken from the request that produced it: the output has no gold-label field, and the
texts are scored by the teacher like any other unlabelled data.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import torch

from .task import Task

REQUEST = """Below is the instruction of a text classification task. Write {n} different, realistic texts of
the kind it classifies, each of which belongs to the category "{name}"{description}. Vary the wording, length
and details. Write one text per line, with no numbering, bullets or quotes.

Instruction:
{instruction}"""


def request(task: Task, label: int, n: int) -> str:
    desc = task.descriptions[label] if task.descriptions else None
    return REQUEST.format(
        n=n,
        name=task.verbalisations[label],
        description=f" ({desc})" if desc else "",
        instruction=task.render("[the text]"),
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


def synthesise(
    task: Task,
    generate: Callable[[str], str],
    per_label: int,
    out: Path,
    batch: int = 10,
    max_rounds: int | None = None,
    log: Callable[[str], None] = print,
) -> int:
    """Ask for ``batch`` texts per request until each label has ``per_label`` new distinct texts, or until
    ``max_rounds`` requests were made for it (by default three times as many as would be needed). Writes JSON
    lines with the task's text field and a ``prompted_label`` note. Returns how many texts were written."""
    seen: set[str] = set()
    written = 0
    rounds = max_rounds or 3 * -(-per_label // batch)
    with out.open("w", encoding="utf-8") as f:
        for li, name in enumerate(task.labels):
            got = 0
            for _ in range(rounds):
                if got >= per_label:
                    break
                for t in parse_lines(generate(request(task, li, min(batch, per_label - got))), task):
                    if got >= per_label or t.lower() in seen:
                        continue
                    seen.add(t.lower())
                    f.write(
                        json.dumps({task.text_field: t, "prompted_label": name}, ensure_ascii=False) + "\n"
                    )
                    got += 1
            written += got
            log(f"{name}: {got} texts" + ("" if got >= per_label else f" (wanted {per_label})"))
    return written
