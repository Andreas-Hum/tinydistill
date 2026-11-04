"""Choose the teacher setup on a dev split: every combination of teacher, few-shot examples and prompt is
scored on held-out gold-labelled training texts (never the test set), best first."""

from __future__ import annotations

import dataclasses
import gc
import random
import time
from collections.abc import Callable

import torch

from .evaluate import accuracy, macro_f1, multilabel_metrics
from .student import few_shot_examples
from .synth import generator
from .task import Task
from .teacher import LabelScorer, load_teacher

PLACEHOLDERS = ("{labels}", "{label}", "{text}", "{examples}")
PROPOSE_TOKENS = 256  # new tokens per rewrite: an instruction is short (the label list is a placeholder)

PROPOSE = """Below is the instruction of a text classification task, given to a language model. Rewrite it
so that the model classifies more accurately: clearer and more precise, with the same task and categories.
Keep the placeholders {placeholders} exactly as written, each once: they are filled in later. Reply with the
rewritten instruction only.

Instruction:
{instruction}"""

REFLECT = """Below is the instruction of a text classification task, given to a language model, and texts it
classified wrongly with it. Rewrite the instruction so that the model gets texts like these right: clearer and
more precise, with the same task and categories. Do not quote the texts. Keep the placeholders {placeholders}
exactly as written, each once: they are filled in later. Reply with the rewritten instruction only.

Mistakes made with it:
{mistakes}

Instruction:
{instruction}"""
MISTAKES = 8  # misclassified dev texts shown per reflection request
SHORT = 300  # characters of each shown text


def _clean(reply: str) -> str:
    """A reply without surrounding code fences, quotes or an "Instruction:" heading."""
    t = reply.strip().strip("`").strip()
    if t.lower().startswith(("text\n", "yaml\n")):  # the language tag of a code fence
        t = t.split("\n", 1)[1]
    if t.lower().startswith("instruction:"):
        t = t[len("instruction:") :]
    return t.strip().strip('"“”').strip()


def propose(
    task: Task,
    generate: Callable[[str], str],
    n: int,
    log: Callable[[str], None] = print,
    mistakes: str = "",
    seen: set[str] | None = None,
) -> list[tuple[str, Task]]:
    """Candidate task files: ``n`` rewrites of the instruction by the model behind ``generate`` (one request
    each), keeping those with every placeholder of the original exactly once and no other, then the task with
    label names only (when it has descriptions). Named "1", "2", ... and "names-only".

    ``mistakes``: texts the instruction got wrong (see ``_mistake``); the request asks for rewrites that fix
    them, and no names-only task is added. ``seen``: instructions already tried, skipped and added to."""
    keep = [p for p in PLACEHOLDERS if p in task.prompt]
    template = REFLECT if mistakes else PROPOSE
    message = template.format(
        placeholders=", ".join(keep), instruction=task.prompt.strip(), mistakes=mistakes
    )
    seen = set() if seen is None else seen
    seen.add(task.prompt.strip())
    out: list[tuple[str, Task]] = []
    for _ in range(n):
        prompt = _clean(generate(message))
        if prompt in seen or any(prompt.count(p) != int(p in keep) for p in PLACEHOLDERS):
            continue
        if task.multi_label and prompt.index("{label}") < prompt.index("{text}"):
            continue  # as load_task demands
        candidate = dataclasses.replace(task, prompt=prompt)
        try:
            candidate.render("text", [("example", 0)] if "{examples}" in prompt else None, "label")
        except Exception:
            continue
        seen.add(prompt)
        out.append((str(len(out) + 1), candidate))
    log(f"proposed {n} rewrites of the instruction: kept {len(out)}, dropped {n - len(out)}")
    if task.descriptions and not mistakes:
        out.append(("names-only", dataclasses.replace(task, descriptions=[])))
    return out


def dev_split(texts: list[str], gold: list, size: int, seed: int) -> tuple[list[int], list[int]]:
    """(dev indices, the rest), among texts that have a gold label."""
    labelled = [i for i, g in enumerate(gold) if g not in (None, [])]
    rng = random.Random(seed + 1)  # not the stream that picks few-shot examples
    dev = sorted(rng.sample(labelled, min(size, len(labelled))))
    chosen = set(dev)
    return dev, [i for i in range(len(texts)) if i not in chosen]


def _mistake(task: Task, text: str, gold, pred) -> str:
    """One misclassified dev text for a reflection request, shortened."""

    def name(x) -> str:
        if task.multi_label:
            return ", ".join(task.verbalisations[i] for i in sorted(x)) or "none"
        return task.verbalisations[x]

    text = " ".join(text.split())
    text = text if len(text) <= SHORT else text[:SHORT] + "..."
    return f"Text: {text}\nCorrect answer: {name(gold)}\nThe model's answer: {name(pred)}"


def tune(
    tasks: list[tuple[str, Task]],
    texts: list[str],
    gold: list,
    teachers: list[str],
    shots: list[int],
    dev_size: int,
    device: str,
    load_4bit: bool,
    seed: int,
    log: Callable[[str], None],
    n_propose: int = 0,
    prefix: str = "",
    rounds: int = 1,
) -> list[dict]:
    """One row per (teacher, prompt, shots): dev accuracy, macro-F1, seconds per text and the instruction.
    ``n_propose`` N: the first teacher also writes N rewrites of the first task's instruction (see
    ``propose``); they are appended to ``tasks``, named ``<prefix><name>.yaml``, and scored too.

    ``rounds`` R > 1 (with ``n_propose``): the dev split is cut in two halves (seeded). Every score is on one
    half only; the other half's mistakes of the best instruction so far are shown to the teacher, which writes
    N rewrites of it that should fix them (round 2 is named ``<prefix>r2-<k>.yaml``, and so on). Stops after R
    rounds or the first round that brings no improvement. Rows then also have the round."""
    dev, rest = dev_split(texts, gold, dev_size, seed)
    if not dev:
        raise SystemExit("tune needs gold labels in the training data")
    if rounds > 1 and not n_propose:
        raise SystemExit("--rounds needs --propose N")
    # rewrites keep the first task's placeholders, so these settings are all there is to score
    if not any(k == 0 or (not t.multi_label and "{examples}" in t.prompt) for _, t in tasks for k in shots):
        raise SystemExit("nothing to score: --few-shot values above 0 need an {examples} line in the prompt")
    # the shown half goes last: rows score dev[:n], the teacher sees mistakes in dev[n:]
    shown = set(random.Random(seed + 2).sample(dev, len(dev) // 2)) if rounds > 1 else set()
    dev = [i for i in dev if i not in shown] + sorted(shown)
    n = len(dev) - len(shown)
    dev_texts, dev_gold = [texts[i] for i in dev], [gold[i] for i in dev]
    rows: list[dict] = []
    wrong: dict[tuple[str, int], list[str]] = {}  # (prompt, shots) -> its mistakes on the shown half
    seen: set[str] = set()
    round_of: dict[str, int] = {}  # the candidates of round 2 on; the rest are round 1
    for ti, name in enumerate(teachers):  # one teacher in memory at a time
        log(f"loading teacher {name}")
        base = load_teacher(name, tasks[0][1], device, load_4bit)
        if not isinstance(base, LabelScorer):
            raise SystemExit("tune needs a local teacher (not --vllm-url or --api-url)")
        if n_propose and ti == 0:
            torch.manual_seed(seed)  # sampled, so seeded
            generate = generator(base.model, base.tok, max_new_tokens=PROPOSE_TOKENS)
            new = propose(tasks[0][1], generate, n_propose, log, seen=seen)
            tasks.extend((f"{prefix}{k}.yaml", t) for k, t in new)
        r = 1
        for j, (prompt, task) in enumerate(tasks):  # a later round appends to tasks while they are scored
            for k in shots:
                if k and (task.multi_label or "{examples}" not in task.prompt):
                    log(f"skipping {k}-shot with {prompt}: the prompt has no {{examples}} line")
                    continue
                examples = (
                    few_shot_examples([texts[i] for i in rest], [gold[i] for i in rest], k, seed)
                    if k
                    else None
                )
                scorer = LabelScorer(
                    base.model, base.tok, task, max_tokens=base.max_tokens, examples=examples
                )
                t0 = time.perf_counter()
                scores = scorer.score_many(dev_texts)
                seconds = (time.perf_counter() - t0) / len(dev_texts)
                if task.multi_label:
                    pred: list = [frozenset(i for i, x in enumerate(s) if x > 0) for s in scores]
                    golds: list = [frozenset(g) for g in dev_gold]
                    m = multilabel_metrics(pred[:n], golds[:n], len(task.labels))
                else:
                    pred, golds = [max(range(len(s)), key=s.__getitem__) for s in scores], dev_gold
                    m = {"accuracy": accuracy(pred[:n], golds[:n]), "macro_f1": macro_f1(pred[:n], golds[:n])}
                wrong[(prompt, k)] = [
                    _mistake(task, t, g, p)
                    for t, g, p in zip(dev_texts[n:], golds[n:], pred[n:], strict=True)
                    if p != g
                ]
                row = {
                    "teacher": name,
                    "prompt": prompt,
                    "few_shot": k,
                    **m,
                    "seconds_per_text": seconds,
                    "instruction": task.prompt,
                    "descriptions": bool(task.descriptions),
                    **({"round": round_of.get(prompt, 1)} if rounds > 1 else {}),
                }
                log(
                    f"{name} | {prompt} | {k}-shot: "
                    f"accuracy {row['accuracy']:.3f}, macro-F1 {row['macro_f1']:.3f}"
                )
                rows.append(row)
            if ti == 0 and r < rounds and j == len(tasks) - 1:  # the round's last candidate is scored
                best = min(rows, key=lambda x: (-x["accuracy"], -x["macro_f1"]))  # the earlier one on a tie
                mistakes = wrong[(best["prompt"], best["few_shot"])]
                if r > 1 and round_of.get(best["prompt"], 1) < r:
                    log(f"round {r} brought no improvement: stopping")
                elif not mistakes:
                    log("the best instruction makes no mistakes on the shown half: stopping")
                else:
                    r += 1
                    shown_mistakes = mistakes[:MISTAKES]
                    log(f"round {r}: rewriting {best['prompt']} with {len(shown_mistakes)} of its mistakes")
                    feedback = "\n\n".join(shown_mistakes)
                    new = propose(dict(tasks)[best["prompt"]], generate, n_propose, log, feedback, seen)
                    tasks.extend((f"{prefix}r{r}-{k}.yaml", t) for k, t in new)
                    round_of.update((f"{prefix}r{r}-{k}.yaml", r) for k, _ in new)
        del base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return sorted(rows, key=lambda r: (-r["accuracy"], -r["macro_f1"]))


def table(rows: list[dict], n_dev: int) -> str:
    head = (
        "| Teacher | Prompt | Few-shot examples per class | Dev accuracy | Dev macro-F1 "
        "| Seconds per text |\n"
        "|---|---|---|---|---|---|\n"
    )
    body = "\n".join(
        f"| `{r['teacher']}` | {r['prompt']} | {r['few_shot']} "
        f"| {r['accuracy']:.3f} | {r['macro_f1']:.3f} "
        f"| {r['seconds_per_text']:.3f} |"
        for r in rows
    )
    return (
        head
        + body
        + (
            f"\n\nScored on {n_dev} gold-labelled training texts held out from the few-shot examples "
            "(not the test set), best first.\n"
        )
    )


def instructions(rows: list[dict], tasks: list[tuple[str, Task]]) -> str:
    """Every prompt's instruction with its best dev accuracy, for the report."""
    best: dict[str, dict] = {}
    for r in rows:  # best first
        best.setdefault(r["prompt"], r)
    parts = [
        f"`{name}`"
        + (f" (round {best[name]['round']})" if "round" in best[name] else "")
        + f": dev accuracy {best[name]['accuracy']:.3f}"
        + ("" if t.descriptions else ", label names only")
        + f"\n\n```text\n{t.prompt.strip()}\n```\n"
        for name, t in tasks
        if name in best
    ]
    return "\n### Instructions\n\n" + "\n".join(parts)
