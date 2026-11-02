"""Task definition: label names, their verbalisations and the instruction prompt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Task:
    name: str
    labels: list[str]  # label names, in a fixed order (class index = position)
    verbalisations: list[str]  # what the teacher is asked to say for each label
    prompt: str  # must contain {text}; may contain {labels}
    text_field: str = "text"
    label_field: str | None = "label"

    def render(self, text: str) -> str:
        """The user message for one input text."""
        options = "\n".join(f"- {v}" for v in self.verbalisations)
        return self.prompt.replace("{labels}", options).replace("{text}", text)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "text_field": self.text_field,
            "label_field": self.label_field,
            "prompt": self.prompt,
            "labels": dict(zip(self.labels, self.verbalisations, strict=True)),
        }


def load_task(path: str | Path) -> Task:
    """Load a task YAML file.

    ``labels`` is either a mapping ``name: verbalisation`` or a list of names (each name is then its own
    verbalisation).
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    labels = raw.get("labels")
    if isinstance(labels, dict):
        names = [str(k) for k in labels]
        verbs = [str(v) if v is not None else str(k) for k, v in labels.items()]
    elif isinstance(labels, list):
        names = verbs = [str(x) for x in labels]
    else:
        raise ValueError(f"{path}: 'labels' must be a mapping or a list")
    if len(names) < 2:
        raise ValueError(f"{path}: need at least two labels")
    if len(set(verbs)) != len(verbs):
        raise ValueError(f"{path}: verbalisations must be unique")
    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or prompt.count("{text}") != 1:
        raise ValueError(f"{path}: 'prompt' must be a string containing {{text}} exactly once")
    return Task(
        name=str(raw.get("name", Path(path).stem)),
        labels=names,
        verbalisations=verbs,
        prompt=prompt,
        text_field=str(raw.get("text_field", "text")),
        label_field=raw.get("label_field", "label"),
    )
