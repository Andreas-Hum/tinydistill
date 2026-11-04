"""Task definition: label ids, their natural-language names (what the teacher scores), optional descriptions
and the instruction prompt."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SURFACE_FORMS = ("lowercase", "capitalised", "uppercase")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # option letters of a lettered prompt


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
    ordinal: bool = False  # the labels are listed in order (e.g. 1 to 5 stars): also report distances
    multi_label: bool = False  # each label is a separate yes/no question ({label} in the prompt)
    answers: tuple[str, str] = ("yes", "no")  # the teacher's answers to a multi-label question

    def render(
        self,
        text: str,
        examples: list[tuple[str, int]] | None = None,
        label: str | None = None,
        lettered: bool = False,
    ) -> str:
        """The user message for one input text. ``examples`` are (text, label index) pairs shown in place of
        {examples}; without them the {examples} line is removed. ``label`` fills {label} (multi-label).
        ``lettered``: the labels are options A, B, C, ... (examples answer with the letter) and the message
        ends with an instruction to answer with the letter (for hosted API teachers)."""
        descs = self.descriptions or [None] * len(self.labels)
        marks = [f"{c}." for c in LETTERS[: len(self.labels)]] if lettered else ["-"] * len(self.labels)
        options = "\n".join(
            f"{m} {v}: {d}" if d else f"{m} {v}"
            for m, v, d in zip(marks, self.verbalisations, descs, strict=True)
        )
        prompt = self.prompt.replace("{labels}", options)
        if examples:
            answer = LETTERS if lettered else self.verbalisations
            shots = "\n".join(f'- "{" ".join(t.split())}" -> {answer[y]}' for t, y in examples)
            prompt = prompt.replace("{examples}", "\nExamples:\n" + shots)
        else:
            prompt = prompt.replace("{examples}\n", "").replace("{examples}", "")
        if label is not None:
            prompt = prompt.replace("{label}", label)
        prompt = prompt.replace("{text}", text)
        return prompt + "\nAnswer with the letter of one option." if lettered else prompt

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
            "ordinal": self.ordinal,
            "multi_label": self.multi_label,
            "answers": list(self.answers),
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
    elif isinstance(
        labels, list
    ):  # ids, or one-key mappings `- id: name` / `- id: {name: ..., description: ...}`
        items = []
        for k in labels:
            if isinstance(k, dict):
                if len(k) != 1:
                    raise ValueError(f"{path}: a label list item must be an id or a single `id: ...` mapping")
                items.append(next((str(i), v) for i, v in k.items()))
            else:
                items.append((str(k), None))
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
    multi = bool(raw.get("multi_label", False))
    if not isinstance(prompt, str) or prompt.count("{text}") != 1:
        raise ValueError(f"{path}: 'prompt' must contain {{text}} exactly once")
    if multi:
        if prompt.count("{label}") != 1 or prompt.index("{label}") < prompt.index("{text}"):
            raise ValueError(f"{path}: a multi-label prompt must contain {{label}} once, after {{text}}")
        if "{examples}" in prompt:
            raise ValueError(f"{path}: few-shot examples are not supported for multi-label tasks")
    elif "{labels}" not in prompt:
        raise ValueError(f"{path}: 'prompt' must contain {{labels}} and contain {{text}} exactly once")
    answers = raw.get("answers", ["yes", "no"])
    if not (
        isinstance(answers, list) and len(answers) == 2 and all(isinstance(a, str) and a for a in answers)
    ):
        raise ValueError(f"{path}: 'answers' must be two strings, the positive answer first")
    return Task(
        name=str(raw.get("name", Path(path).stem)),
        labels=[lid for lid, _ in items],
        verbalisations=verbs,
        prompt=prompt,
        text_field=str(raw.get("text_field", "text")),
        label_field=raw.get("label_field", "label"),
        descriptions=descs if any(descs) else [],
        surface_forms=_parse_forms(path, raw.get("surface_forms", list(SURFACE_FORMS))),
        ordinal=bool(raw.get("ordinal", False)),
        multi_label=multi,
        answers=(answers[0], answers[1]),
    )


DEFAULT_PROMPT = """Classify the text into exactly one of these categories:
{labels}
{examples}

Text: {text}

Reply with the category exactly as written in the list."""


def task_yaml_from_data(
    spec: str, text_field: str = "text", label_field: str = "label", scan: int = 50_000
) -> str:
    """A starter task file for a dataset: its label names (from a Hugging Face ClassLabel feature, or the
    distinct string labels in the first `scan` rows of a CSV/JSONL file) and a generic prompt to edit."""
    from itertools import islice

    from .data import _rows

    rows, class_names = _rows(spec)
    names = (class_names or {}).get(label_field)
    if names is None:
        values = {row[label_field] for row in islice(rows, scan) if row.get(label_field) not in (None, "")}
        if not values:
            raise ValueError(f"{spec}: no values in field {label_field!r}")
        if any(isinstance(v, int) or str(v).isdigit() for v in values):
            raise ValueError(f"{spec}: the labels are numbers without names; write the label list by hand")
        names = sorted(str(v) for v in values)
    name = re.sub(r"[^A-Za-z0-9]+", "-", spec.split(":")[1] if spec.startswith("hf:") else Path(spec).stem)
    body = {
        "name": name.strip("-").lower(),
        "text_field": text_field,
        "label_field": label_field,
        "prompt": DEFAULT_PROMPT,
        "labels": list(names),
    }

    class Literal(str):
        pass

    def literal(dumper, data):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")

    yaml.add_representer(Literal, literal, Dumper=yaml.SafeDumper)
    body["prompt"] = Literal(DEFAULT_PROMPT)
    header = (
        f"# Task file for {spec}, written by `tinydistill init`. Edit the prompt and, if needed, the label\n"
        "# names (id: name) or add descriptions (id: {description: ...}); see the README.\n"
    )
    return header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=120)
