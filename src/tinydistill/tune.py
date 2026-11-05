"""Choose the teacher setup on a dev split: every combination of teacher, few-shot examples and prompt is
scored on held-out gold-labelled training texts (never the test set), best first."""

from __future__ import annotations

import dataclasses
import gc
import random
import re
import time
from collections.abc import Callable

import torch

from .evaluate import accuracy, macro_f1, multilabel_metrics
from .student import few_shot_examples
from .synth import generator
from .task import Task
from .teacher import LabelScorer, load_teacher

DESCRIBED = {  # each placeholder as the teacher sees it in a rewrite request
    "{labels}": "[the list of categories]",
    "{label}": "[the category asked about]",
    "{text}": "[the text to classify]",
    "{examples}": "[labelled examples]",
}
PLACEHOLDERS = tuple(DESCRIBED)
PROPOSE_TOKENS = 256  # new tokens per rewrite: an instruction is short (the label list is a placeholder)
LEAD_CHARS = 600  # longest kept rewrite of the instruction (or twice the original's length, if more)

PROPOSE = """Below is a prompt of a text classification task, given to a language model. The parts in square
brackets are filled in later. Rewrite the instruction at its start (repeated below) so that the model
classifies more accurately: clearer and more precise, with the same task and categories. The rest of the
prompt stays as it is, so do not list the categories or add the text. Reply with the new instruction only.

Prompt:
{prompt}

Instruction to rewrite:
{lead}"""

REFLECT = """Below is a prompt of a text classification task, given to a language model, and texts it
classified wrongly with it. The parts in square brackets are filled in later. Rewrite the instruction at its
start (repeated below) so that the model gets texts like these right: clearer and more precise, with the same
task and categories. Do not quote the texts. The rest of the prompt stays as it is, so do not list the
categories or add the text. Reply with the new instruction only.

Mistakes made with it:
{mistakes}

Prompt:
{prompt}

Instruction to rewrite:
{lead}"""

# for a prompt that starts with a placeholder: the teacher rewrites all of it
PROPOSE_WHOLE = """Below is the instruction of a text classification task, given to a language model.
Rewrite it so that the model classifies more accurately: clearer and more precise, with the same task and
categories. Keep the placeholders {placeholders} exactly as written, each once: they are filled in later.
Reply with the rewritten instruction only.

Instruction:
{instruction}"""

REFLECT_WHOLE = """Below is the instruction of a text classification task, given to a language model, and
texts it classified wrongly with it. Rewrite the instruction so that the model gets texts like these right:
clearer and more precise, with the same task and categories. Do not quote the texts. Keep the placeholders
{placeholders} exactly as written, each once: they are filled in later. Reply with the rewritten instruction
only.

Mistakes made with it:
{mistakes}

Instruction:
{instruction}"""
MISTAKES = 8  # misclassified dev texts shown per reflection request
SHORT = 300  # characters of each shown text


def _clean(reply: str) -> str:
    """A reply without surrounding code fences, quotes or a heading such as "Instruction:" or "Rewritten
    instruction:" (on a line of its own or not)."""
    t = reply.strip().strip("`").strip()
    if t.lower().startswith(("text\n", "yaml\n")):  # the language tag of a code fence
        t = t.split("\n", 1)[1]
    t = re.sub(r"(?i)^[^\n:]*instruction[^\n:]*:", "", t.strip())
    return t.strip().strip('"“”').strip()


def split(prompt: str) -> tuple[str, str]:
    """(lead, rest): the text before the prompt's first placeholder, and the rest from there, which rewrites
    keep as written. A one-word heading line just before the placeholder (such as "Options:"), after other
    lines, is part of the rest. The lead is empty when the prompt starts with a placeholder."""
    at = min((prompt.find(p) for p in PLACEHOLDERS if p in prompt), default=len(prompt))
    heading = re.search(r"\n(\w+:\s*)\Z", prompt[:at])
    at = heading.start(1) if heading else at
    return prompt[:at], prompt[at:]


def _lead_problem(task: Task, reply: str, longest: int, mistakes: str) -> str:
    """Why a rewrite of the lead is dropped, or "" to keep it."""
    if not reply:
        return "empty reply"
    if len(reply) > longest:
        return f"reply longer than {longest} characters"
    if re.search(r"\{\w+\}|\[[^\]\n]*\]", reply):  # also one in square brackets, as the request shows them
        return "reply contains a placeholder"
    if all(v.lower() in reply.lower() for v in task.verbalisations):
        return "reply contains the whole option list"
    shown = [x[len("Text: ") :].removesuffix("...") for x in mistakes.splitlines() if x.startswith("Text: ")]
    if re.search(r"(?im)^(text|correct answer|the model's answer):", reply) or any(x in reply for x in shown):
        return "reply quotes the mistakes"
    return ""


def _whole_problem(task: Task, prompt: str, keep: list[str]) -> str:
    """Why a rewrite of the whole prompt is dropped, or "" to keep it."""
    if any(prompt.count(p) != int(p in keep) for p in PLACEHOLDERS):
        return "reply does not have the placeholders " + ", ".join(keep) + ", each once"
    if task.multi_label and prompt.index("{label}") < prompt.index("{text}"):
        return "reply has {label} before {text}"  # as load_task demands
    try:
        dataclasses.replace(task, prompt=prompt).render(
            "text", [("example", 0)] if "{examples}" in prompt else None, "label"
        )
    except Exception:
        return "reply does not render"
    return ""


def propose(
    task: Task,
    generate: Callable[[str], str],
    n: int,
    log: Callable[[str], None] = print,
    mistakes: str = "",
    seen: set[str] | None = None,
) -> list[tuple[str, Task]]:
    """Candidate task files: ``n`` rewrites by the model behind ``generate`` (one request each) of the
    instruction before the first placeholder (see ``split``), each followed by the rest of the prompt as
    written, then the task with label names only (when it has descriptions). Named "1", "2", ... and
    "names-only". The request shows the placeholders as descriptions in square brackets; a reply that is
    empty, too long, has a placeholder or the whole option list, or quotes the mistakes is dropped, and the
    reason logged. A prompt that starts with a placeholder is rewritten whole instead, keeping a reply only
    with every placeholder of the original exactly once and no other.

    ``mistakes``: texts the instruction got wrong (see ``_mistake``); the request asks for rewrites that fix
    them, and no names-only task is added. ``seen``: instructions already tried, skipped and added to."""
    lead, rest = split(task.prompt)
    keep = [p for p in PLACEHOLDERS if p in task.prompt]
    if lead.strip():
        shown = task.prompt
        for p, d in DESCRIBED.items():
            shown = shown.replace(p, d)
        message = (REFLECT if mistakes else PROPOSE).format(
            prompt=shown.strip(), lead=lead.strip(), mistakes=mistakes
        )
        sep = lead[len(lead.rstrip()) :]  # the line break (or space) between the lead and the rest
        longest = max(LEAD_CHARS, 2 * len(lead.strip()))
    else:
        log("the prompt starts with a placeholder: the teacher rewrites the whole prompt")
        message = (REFLECT_WHOLE if mistakes else PROPOSE_WHOLE).format(
            placeholders=", ".join(keep), instruction=task.prompt.strip(), mistakes=mistakes
        )
    seen = set() if seen is None else seen
    seen.add(task.prompt.strip())
    out: list[tuple[str, Task]] = []
    for _ in range(n):
        reply = _clean(generate(message))
        if lead.strip():
            prompt = reply + sep + rest
            why = _lead_problem(task, reply, longest, mistakes)
        else:
            prompt = reply
            why = _whole_problem(task, prompt, keep)
        if not why and prompt.strip() in seen:
            why = "an instruction already tried"
        if why:
            log(f"dropped: {why}")
            continue
        seen.add(prompt.strip())
        out.append((str(len(out) + 1), dataclasses.replace(task, prompt=prompt)))
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
    note = (
        "The teacher's rewrites change only the instruction before the first placeholder; the rest of the "
        "prompt is kept as written.\n\n"
        if split(tasks[0][1].prompt)[0].strip()
        else ""
    )
    return "\n### Instructions\n\n" + note + "\n".join(parts)
