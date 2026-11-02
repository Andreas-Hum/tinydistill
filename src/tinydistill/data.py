"""Read texts (and optional gold labels) from CSV, JSONL or a Hugging Face dataset, as a stream."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from .task import Task


@dataclass
class Example:
    text: str
    gold: int | None = None  # index into Task.labels


def _gold_index(value: object, task: Task, names: list[str] | None) -> int | None:
    """Map a raw label value to a class index: label names by name, integers by position."""
    if value is None or value == "":
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        value = names[value] if names else value  # HF ClassLabel: int -> name first
        if isinstance(value, int):
            if not 0 <= value < len(task.labels):
                raise ValueError(f"label index {value} out of range")
            return value
    s = str(value)
    if s in task.labels:
        return task.labels.index(s)
    if s in task.verbalisations:
        return task.verbalisations.index(s)
    if s.isdigit() and int(s) < len(task.labels):
        return int(s)
    raise ValueError(f"unknown label {s!r}; expected one of the task's label names")


def _rows(spec: str) -> tuple[Iterator[dict], dict[str, list[str]] | None]:
    if spec.startswith("hf:"):
        # hf:<dataset>[:<config>]:<split>
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise SystemExit("Reading Hugging Face datasets needs: pip install 'tinydistill[hf]'") from e
        parts = spec[3:].split(":")
        if len(parts) == 2:
            name, config, split = parts[0], None, parts[1]
        elif len(parts) == 3:
            name, config, split = parts
        else:
            raise ValueError(f"bad dataset spec {spec!r}; use hf:<dataset>[:<config>]:<split>")
        ds = load_dataset(name, config, split=split, streaming=True)
        return iter(ds), None if ds.features is None else _class_names(ds.features)
    path = Path(spec)
    if path.suffix == ".csv":

        def gen() -> Iterator[dict]:
            with path.open(newline="", encoding="utf-8") as f:
                yield from csv.DictReader(f)

        return gen(), None
    if path.suffix in (".jsonl", ".json"):

        def gen() -> Iterator[dict]:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

        return gen(), None
    raise ValueError(f"unsupported data file {spec!r}; use .csv, .jsonl or hf:<dataset>:<split>")


def _class_names(features: dict) -> dict[str, list[str]]:
    return {k: list(f.names) for k, f in features.items() if hasattr(f, "names")}


def read_examples(
    spec: str, task: Task, limit: int | None = None, with_gold: bool = True
) -> Iterator[Example]:
    rows, class_names = _rows(spec)
    names = class_names.get(task.label_field or "") if class_names else None
    for row in islice(rows, limit):
        if task.text_field not in row:
            raise KeyError(f"{spec}: no field {task.text_field!r} (set text_field in the task file)")
        gold = None
        if with_gold and task.label_field and task.label_field in row:
            gold = _gold_index(row[task.label_field], task, names)
        yield Example(str(row[task.text_field]), gold)
