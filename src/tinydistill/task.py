"""Task definition: label ids, their natural-language names (what the teacher scores), optional descriptions
and the instruction prompt."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SURFACE_FORMS = ("lowercase", "capitalised", "uppercase")


def label_name(label_id: str) -> str:
    """Natural-language name for a label id: card_arrival -> 'card arrival', top-up -> 'top up'."""
    return re.sub(r"[\s_\-.]+", " ", label_id).strip()


@dataclass(frozen=True)
class Task:
    name: str
    labels: list[str]  # label ids, in a fixed order (class index = position)
    verbalisations: list[str]  # natural-language label names; the teacher scores these
    prompt: str  # must contain {labels} and {text}; may contain {examples} (few-shot examples)
    text_field: str = "text"
    label_field: str | None = "label"
    descriptions: list[str | None] = field(default_factory=list)  # optional, shown in the prompt only
    # Extra ways of writing each name that the teacher's score includes (the name as written always counts).
    surface_forms: tuple[str, ...] = SURFACE_FORMS

    def render(self, text: str, examples: list[tuple[str, int]] | None = None) -> str:
        """The user message for one input text. ``examples`` are (text, label index) pairs shown in place of
        {examples}; without them the {examples} line is removed."""
        descs = self.descriptions or [None] * len(self.labels)
        options = "\n".join(
            f"- {v}: {d}" if d else f"- {v}" for v, d in zip(self.verbalisations, descs, strict=True)
        )
        prompt = self.prompt.replace("{labels}", options)
        if examples:
            shots = "\n".join(f'- "{" ".join(t.split())}" -> {self.verbalisations[y]}' for t, y in examples)
            prompt = prompt.replace("{examples}", "\nExamples:\n" + shots)
        else:
            prompt = prompt.replace("{examples}\n", "").replace("{examples}", "")
        return prompt.replace("{text}", text)

    def to_dict(self) -> dict:
        labels = {}
        for i, (lid, v) in enumerate(zip(self.labels, self.verbalisations, strict=True)):
            d = self.descriptions[i] if self.descriptions else None
            labels[lid] = {"name": v, "description": d} if d else v
        return {
            "name": self.name,
            "text_field": self.text_field,
            "label_field": self.label_field,
            "prompt": self.prompt,
            "surface_forms": list(self.surface_forms),
            "labels": labels,
        }


def _parse_label(path, lid: str, value) -> tuple[str, str | None]:
    if value is None:
        return label_name(lid), None
    if isinstance(value, str):
        return value, None
    if isinstance(value, dict):
        unknown = set(value) - {"name", "description"}
        if unknown:
            raise ValueError(f"{path}: label {lid!r} has unknown keys {sorted(unknown)}")
        name = value.get("name") or label_name(lid)
        desc = value.get("description")
        return str(name), (str(desc) if desc else None)
    raise ValueError(f"{path}: label {lid!r} must be a name, a mapping with name/description, or empty")


def _parse_forms(path, value) -> tuple[str, ...]:
    if not isinstance(value, list) or not set(value) <= set(SURFACE_FORMS):
        raise ValueError(f"{path}: 'surface_forms' must be a list drawn from {list(SURFACE_FORMS)}")
    return tuple(f for f in SURFACE_FORMS if f in value)


def load_task(path: str | Path) -> Task:
    """Load a task YAML file. ``labels`` is a list of label ids, or a mapping from id to either a name or
    ``{name: ..., description: ...}`` (both optional). A missing name is derived from the id."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    labels = raw.get("labels")
    if isinstance(labels, dict):
        items = [(str(k), v) for k, v in labels.items()]
    elif isinstance(labels, list):
        items = [(str(k), None) for k in labels]
    else:
        raise ValueError(f"{path}: 'labels' must be a mapping or a list")
    if len(items) < 2:
        raise ValueError(f"{path}: need at least two labels")
    parsed = [_parse_label(path, lid, v) for lid, v in items]
    verbs = [p[0] for p in parsed]
    descs = [p[1] for p in parsed]
    if len(set(verbs)) != len(verbs):
        raise ValueError(f"{path}: label names must be unique")
    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or prompt.count("{text}") != 1 or "{labels}" not in prompt:
        raise ValueError(f"{path}: 'prompt' must contain {{labels}} and contain {{text}} exactly once")
    return Task(
        name=str(raw.get("name", Path(path).stem)),
        labels=[lid for lid, _ in items],
        verbalisations=verbs,
        prompt=prompt,
        text_field=str(raw.get("text_field", "text")),
        label_field=raw.get("label_field", "label"),
        descriptions=descs if any(descs) else [],
        surface_forms=_parse_forms(path, raw.get("surface_forms", list(SURFACE_FORMS))),
    )
