"""Read texts (and optional gold labels) from CSV, JSONL or a Hugging Face dataset, as a stream."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from .task import Task


@dataclass
class Example:
    text: str
    gold: int | list[int] | None = None  # index into Task.labels; a sorted list of indices for multi-label


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


def _gold_set(value: object, task: Task, names: list[str] | None) -> list[int] | None:
    """Multi-label gold: a list of label names or indices, or a string of them separated by "|"."""
    if value is None or value == "":
        return None
    items = value if isinstance(value, list | tuple) else str(value).split("|")
    found = {_gold_index(v.strip() if isinstance(v, str) else v, task, names) for v in items}
    return sorted(i for i in found if i is not None)


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
    out = {}
    for k, f in features.items():
        inner: Any = getattr(f, "feature", None)  # Sequence(ClassLabel) for multi-label datasets
        if hasattr(f, "names") or hasattr(inner, "names"):
            out[k] = list(f.names if hasattr(f, "names") else inner.names)
    return out


def count_rows(spec: str, limit: int | None = None) -> int | None:
    """How many examples a data source has (None if it cannot tell cheaply), capped at ``limit``."""
    n: int | None = None
    if spec.startswith("hf:"):
        try:
            from datasets import load_dataset_builder

            parts = spec[3:].split(":")
            name, config, split = (parts[0], None, parts[1]) if len(parts) == 2 else parts
            n = load_dataset_builder(name, config).info.splits[split].num_examples
        except Exception:
            n = None
    elif Path(spec).suffix in (".jsonl", ".json"):
        with Path(spec).open(encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())
    elif Path(spec).suffix == ".csv":
        with Path(spec).open(newline="", encoding="utf-8") as f:
            n = sum(1 for _ in csv.DictReader(f))
    if n is not None and limit is not None:
        n = min(n, limit)
    return n


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
            parse = _gold_set if task.multi_label else _gold_index
            gold = parse(row[task.label_field], task, names)
        yield Example(str(row[task.text_field]), gold)
