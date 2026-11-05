"""Command-line interface: label -> train -> export -> eval, or `run` for all of them."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import yaml

from .task import Task, load_task

DEFAULT_TEACHER = "Qwen/Qwen2.5-1.5B-Instruct"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _meta(run: Path, update: dict | None = None) -> dict:
    path = run / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if update:
        meta.update(update)
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _argmax(xs) -> int:
    return max(range(len(xs)), key=xs.__getitem__)


# ---------- steps ----------


def _teachers(spec: str) -> list[str]:
    return [t.strip() for t in spec.split(",") if t.strip()]


def _write_task(path: Path, task: Task) -> None:
    path.write_text(yaml.safe_dump(task.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")


def _variants(task: Task, paths: list[str]) -> list[Task]:
    """Other task files for the same task (other prompts): same label ids in the same order and the same
    multi_label setting, so that their label scores can be averaged."""
    variants = [load_task(p) for p in paths]
    for p, v in zip(paths, variants, strict=True):
        if v.labels != task.labels:
            raise SystemExit(
                f"{p} must have the task's label ids in the same order: {', '.join(task.labels)} "
                f"(it has {', '.join(v.labels)})"
            )
        if v.multi_label != task.multi_label:
            raise SystemExit(f"{p} must have the task's multi_label setting ({task.multi_label})")
    return variants


def _run_members(run: Path, meta: dict, task: Task, name: str, device: str, examples) -> list:
    """One teacher of a run as it labelled: a scorer per prompt (the task's, then its variants)."""
    from .teacher import load_teacher, prompt_members

    scorer = load_teacher(
        name,
        task,
        device,
        meta.get("load_4bit", False),
        meta.get("teacher_tokens", 2048),
        examples,
        meta.get("vllm_url"),
        meta.get("shortlist"),
        meta.get("shortlist_model"),
        meta.get("shortlist_keep"),
        meta.get("api_url"),
        meta.get("api_key_env", "OPENAI_API_KEY"),
        meta.get("api_samples"),
        meta.get("api_temperature", 0.7),
        meta.get("few_shot_similar", 0),
        meta.get("few_shot_model"),
        meta.get("reason_tokens", 0) if meta.get("reason") else 0,
    )
    return prompt_members(scorer, [load_task(run / f) for f in meta.get("prompts") or []])


def cmd_label(a: argparse.Namespace, only: set[str] | None = None) -> None:
    """``only``: label just these training texts (run --budget); the test set is labelled in full."""
    import gc

    import torch

    from .data import count_rows, read_examples
    from .progress import Progress
    from .teacher import (
        ScoreCache,
        combine_scores,
        iter_unique,
        load_teacher,
        pick_device,
        prompt_members,
        quick_key,
        text_hash,
    )

    task = load_task(a.task)
    if a.dry_run and not a.api_url:
        raise SystemExit("--dry-run estimates the cost of labelling with --api-url")
    sources = [p.strip() for p in (a.prompts or "").split(",") if p.strip()]
    variants = _variants(task, sources)
    if variants and (a.vllm_url or a.api_url):
        raise SystemExit("--prompts needs a local teacher, not --vllm-url or --api-url")
    reason = a.reason_tokens if a.reason else 0
    if a.reason and a.reason_tokens < 1:
        raise SystemExit("--reason-tokens must be positive")
    if a.few_shot_similar and a.chunk_tokens:
        raise SystemExit(
            "--few-shot-similar cannot be used with --chunk-tokens: a window is not its document's text, so "
            "the document, with its gold label, would be among its examples"
        )
    run = Path(a.out)
    run.mkdir(parents=True, exist_ok=True)
    prompt_files = [f"prompt-{i}.yaml" for i in range(1, len(variants) + 1)]
    examples = None
    flag = "--few-shot" if a.few_shot else "--few-shot-similar"
    for t in (task, *variants):
        if (a.few_shot or a.few_shot_similar) and "{examples}" not in t.prompt:
            raise SystemExit(
                f"{flag} needs an {{examples}} line in the task's prompt (and in each --prompts file)"
            )
        if a.few_shot_similar and t.prompt.index("{examples}") > t.prompt.index("{text}"):
            raise SystemExit("--few-shot-similar needs {examples} before {text} in the task's prompt")
    if a.few_shot or a.few_shot_similar:
        from .student import few_shot_examples

        rows = list(read_examples(a.data, task, a.limit))
        if not any(e.gold is not None for e in rows):
            raise SystemExit(f"{flag} needs gold labels in the training data")
        gold = cast(list[int | None], [e.gold for e in rows])  # single-label: load_task refuses multi-label
        if a.few_shot:
            examples = few_shot_examples([e.text for e in rows], gold, a.few_shot, a.seed)
        else:  # the pool for --few-shot-similar alone: every gold-labelled training text, once
            pairs = zip([e.text for e in rows], gold, strict=True)
            examples = list(dict.fromkeys((t, g) for t, g in pairs if g is not None))
        if a.few_shot_similar:
            log(f"each text's prompt: its {a.few_shot_similar} most similar of {len(examples)} gold examples")
        else:
            log(f"few-shot prompt with {len(examples)} gold examples ({a.few_shot} per class)")
    cache_dir = Path(a.cache_dir) if a.cache_dir else run
    cache_dir.mkdir(parents=True, exist_ok=True)
    splits = [
        (n, spec, lim)
        for n, spec, lim in (("train", a.data, a.limit), ("test", a.test, a.test_limit))
        if spec
    ]
    cut, chunk = _chunking(a)

    def rows_of(split, spec, limit, with_gold=True):
        rows = read_examples(spec, task, limit, with_gold=with_gold)
        return (e for e in rows if e.text in only) if only is not None and split == "train" else rows

    def texts_of(split, spec, limit):  # what the teacher scores: every text, or every window of it
        return (w for e in rows_of(split, spec, limit, False) for w in cut(e.text))

    caches, fingerprints, device, costs = [], [], "cpu", []
    seconds, scored = 0.0, 0
    keys_file = cache_dir / "teacher-keys.json"  # quick key (tokenizer only) -> full fingerprint
    keys = json.loads(keys_file.read_text(encoding="utf-8")) if keys_file.exists() else {}
    for name in _teachers(a.teacher):  # one teacher in memory at a time, one member per prompt
        quick: list[str | None] = [None] * (1 + len(variants))
        if not a.vllm_url and not a.api_url:  # a server's tokenizer and setup are not known locally
            quick = [
                quick_key(
                    name,
                    t,
                    a.device,
                    a.load_4bit,
                    examples,
                    a.shortlist,
                    a.shortlist_model,
                    a.shortlist_keep,
                    a.few_shot_similar,
                    a.few_shot_model,
                    reason,
                )
                for t in (task, *variants)
            ]
            known = [keys.get(q) for q in quick]
            cached = [ScoreCache(cache_dir / f"teacher-{k}.jsonl") for k in known if k]
            if len(cached) == len(known) and all(
                text_hash(t) in c.scores
                for split, spec, limit in splits
                for t in texts_of(split, spec, limit)
                for c in cached
            ):
                log(f"{name}: every text is in the cache, so the teacher is not loaded")
                fingerprints.extend(cast(list[str], known))
                caches.extend(cached)
                device = pick_device(a.device)
                device = "cuda:0" if device == "cuda" else device  # as a loaded model reports it
                continue
        log(f"loading teacher {name}")
        scorer = load_teacher(
            name,
            task,
            a.device,
            a.load_4bit,
            a.teacher_tokens,
            examples,
            a.vllm_url,
            a.shortlist,
            a.shortlist_model,
            a.shortlist_keep,
            a.api_url,
            a.api_key_env,
            a.api_samples,
            a.api_temperature,
            a.few_shot_similar,
            a.few_shot_model,
            reason,
        )
        if reason:
            log(
                f"the teacher reasons in up to {reason} tokens per text before it answers; the reasoning "
                "cannot reuse the cached prompt, so labelling is much slower"
            )
        device = str(scorer.device)
        members = prompt_members(scorer, variants)
        for pi, (member, q) in enumerate(zip(members, quick, strict=True)):
            who = name.split("/")[-1] + (f", prompt {pi}" if variants else "")  # prompt 0: the task's own
            fingerprints.append(member.fingerprint())
            if q:
                keys = json.loads(keys_file.read_text(encoding="utf-8")) if keys_file.exists() else {}
                keys[q] = fingerprints[-1]
                tmp = keys_file.with_suffix(".tmp")
                tmp.write_text(json.dumps(keys, indent=1), encoding="utf-8")
                tmp.replace(keys_file)
            cache = ScoreCache(cache_dir / f"teacher-{fingerprints[-1]}.jsonl")
            if cache.scores:
                log(f"resuming: {len(cache.scores)} texts already scored")
            if a.dry_run:
                todo = iter_unique(
                    t for s in splits for t in texts_of(*s) if text_hash(t) not in cache.scores
                )
                costs.append(_api_cost(member, list(todo), a.api_samples))
                print(_cost_line(who, *costs[-1], a))
                continue
            for split, spec, limit in splits:
                t0 = time.perf_counter()
                # windows (and a budget's texts) are not counted ahead
                total = None if chunk or only is not None else count_rows(spec, limit)
                bar = Progress(f"label {split} ({who})", total, log, run / "status.json")
                texts = iter_unique(texts_of(split, spec, limit))
                n = cache.fill(member, texts, on_read=bar.update)
                bar.finish()
                seconds += time.perf_counter() - t0
                scored += n
                log(f"{split}: {n} texts scored now by {who}, the rest from the cache")
            caches.append(cache)
        del scorer, members
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if a.dry_run:
        if len(costs) > 1:
            n, requests, tokens_in, tokens_out = (sum(c) for c in zip(*costs, strict=True))
            print(_cost_line("total", n, requests, tokens_in, tokens_out, a))
        print(
            "Input tokens are estimated as the prompt's characters / 4. With --api-samples N each text takes "
            "N requests at most; an API that accepts several answers per request takes one and bills the "
            "prompt once. Nothing was sent or scored."
        )
        return

    # the run's files, written only now so that a dry run leaves the run as it was; none left from earlier
    _write_task(run / "task.yaml", task)
    for file, v in zip(prompt_files, variants, strict=True):
        _write_task(run / file, v)
    (run / "examples.jsonl").unlink(missing_ok=True)
    if examples is not None:
        with (run / "examples.jsonl").open("w", encoding="utf-8") as f:
            for text, y in examples:
                f.write(json.dumps({"text": text, "gold": y}) + "\n")
    if not a.test:
        (run / "test.jsonl").unlink(missing_ok=True)
    multi = task.multi_label
    for split, spec, limit in splits:
        n_rows = 0
        with (run / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for doc, e in enumerate(rows_of(split, spec, limit)):
                wins = cut(e.text)
                if split == "test" or len(wins) == 1:  # the document, scored as the mean of its windows
                    pieces = [(e.text, [combine_scores([c.get(w) for w in wins], multi) for c in caches])]
                else:  # the student trains on the windows
                    pieces = [(w, [c.get(w) for c in caches]) for w in wins]
                for text, per in pieces:
                    row: dict[str, Any] = {"text": text, "gold": e.gold, "scores": combine_scores(per, multi)}
                    if len(per) > 1:
                        row["teacher_scores"] = per
                    if chunk and split == "train":
                        row["doc"] = doc  # the window's document: its row number in the data
                    f.write(json.dumps(row) + "\n")
                    n_rows += 1
        log(f"{split}: {n_rows} {'windows' if chunk and split == 'train' else 'texts'}")
    (run / "reasoning.jsonl").unlink(missing_ok=True)
    if reason:  # each scored text's reasoning, one entry per member (for analyse)
        with (run / "reasoning.jsonl").open("w", encoding="utf-8") as f:
            for split, spec, limit in splits:
                for w in iter_unique(texts_of(split, spec, limit)):
                    why = [c.reasoning.get(text_hash(w)) for c in caches]
                    f.write(json.dumps({"split": split, "text": w, "reasoning": why}) + "\n")
    _meta(
        run,
        {
            "teacher": a.teacher,
            "teachers": _teachers(a.teacher),
            # variant task files copied into the run; the members are teachers x (task, *prompts)
            "prompts": prompt_files,
            "prompt_sources": sources,
            "few_shot": a.few_shot,
            "few_shot_similar": a.few_shot_similar,
            "few_shot_model": a.few_shot_model,
            "n_examples": len(examples or []),
            "load_4bit": a.load_4bit,
            "teacher_device": device,
            "fingerprint": fingerprints[0],
            "fingerprints": fingerprints,
            "data": a.data,
            "test": a.test,
            "label_seconds": round(seconds, 1),
            "label_texts_scored": scored,
            "budget": None if only is None else len(only),  # run --budget: the training texts labelled
            "teacher_tokens": a.teacher_tokens,
            "vllm_url": a.vllm_url,
            "api_url": a.api_url,
            "api_key_env": a.api_key_env,  # the variable's name; the key itself is never written
            "api_samples": a.api_samples,
            "api_temperature": a.api_temperature,
            "shortlist": a.shortlist,
            "shortlist_model": a.shortlist_model,
            "shortlist_keep": a.shortlist_keep,
            "reason": a.reason,
            "reason_tokens": a.reason_tokens,
            **({"chunk_tokens": None, "chunk_stride": None, "chunk_tokenizer": None} | chunk),
        },
    )


def _api_cost(member, texts: list[str], samples: int | None) -> tuple[int, int, int, int]:
    """label --dry-run: (texts, requests, input tokens, output tokens) that labelling ``texts`` with the API
    teacher ``member`` takes. Input tokens are the rendered prompts' characters / 4, an approximation (no
    tokenizer of a hosted model is at hand); output tokens are the most the requests ask for."""
    from .teacher import API_SAMPLE_TOKENS

    shots = member.similar(texts) if member.similar is not None and texts else [None] * len(texts)
    chars = sum(len(member._render(t, s)) for t, s in zip(texts, shots, strict=True))
    k = samples or 1
    return (
        len(texts),
        len(texts) * k,
        round(chars / 4) * k,
        len(texts) * k * (API_SAMPLE_TOKENS if samples else 1),
    )


def _cost_line(
    who: str, n: int, requests: int, tokens_in: int, tokens_out: int, a: argparse.Namespace
) -> str:
    line = (
        f"{who}: {n} texts to label, {requests} requests, about {tokens_in:,} input tokens and at most "
        f"{tokens_out:,} output tokens"
    )
    if a.price_in is not None or a.price_out is not None:
        cost = (tokens_in * (a.price_in or 0) + tokens_out * (a.price_out or 0)) / 1e6
        line += f", about {cost:,.4f} at the given prices"
    return line


def _chunking(a: argparse.Namespace) -> tuple[Any, dict]:
    """For label --chunk-tokens: a function cutting a text into its windows (the text alone without the
    flag), and the settings for meta.json, which train stores with the model."""
    if not a.chunk_tokens:
        return (lambda t: [t]), {}
    from transformers import AutoTokenizer

    from .static import encoder_of
    from .student import DEFAULT_STUDENT, windows

    size = a.chunk_tokens
    stride = a.chunk_stride if a.chunk_stride is not None else max(1, size // 2)
    if size < 1 or not 0 < stride <= size:
        raise SystemExit("--chunk-tokens must be positive and --chunk-stride between 1 and --chunk-tokens")
    name = a.chunk_tokenizer or getattr(a, "student", None) or DEFAULT_STUDENT  # `run` has --student
    name = encoder_of(name) or name  # a static student uses its encoder's tokenizer
    tok = AutoTokenizer.from_pretrained(name)
    log(f"texts over {size} tokens ({name}) are cut into windows of {size} tokens, {stride} apart")
    return (lambda t: windows(tok, t, size, stride)), {
        "chunk_tokens": size,
        "chunk_stride": stride,
        "chunk_tokenizer": name,
    }


# Student variants: (directory name, role, row name). The first seed's models live directly in the run
# directory; further seeds go to run/seeds/<seed>/.
VARIANTS = (
    ("student", "student", "student, PyTorch"),
    ("student-kgold", "baseline", "baseline: student on the K gold labels only, no teacher"),
    ("student-gold", "upper bound", "upper bound: student on all gold labels"),
)


def _seed_dirs(run: Path, name: str) -> list[Path]:
    """The first seed's model, then any further seeds' models, for one variant."""
    extra = sorted((run / "seeds").glob(f"*/{name}"), key=lambda p: int(p.parent.name))
    return [p for p in [run / name, *extra] if p.exists()]


def cmd_train(a: argparse.Namespace) -> None:
    import shutil

    import torch

    from .static import encoder_of
    from .student import few_gold, few_gold_sets, train_student
    from .teacher import pick_device

    run = Path(a.run)
    task = load_task(run / "task.yaml")
    rows = _read_jsonl(run / "train.jsonl")
    if not rows:
        raise SystemExit(f"{run}/train.jsonl is missing or empty: run `tinydistill label` first")
    texts = [r["text"] for r in rows]
    all_gold = [r["gold"] for r in rows]
    if (a.gold_per_class or a.upper_bound) and not any(g is not None for g in all_gold):
        raise SystemExit("--gold-per-class and --upper-bound need gold labels in the training data")
    if a.gold_only and not (a.gold_per_class or a.gold_file):
        raise SystemExit("--gold-only needs --gold-per-class K or --gold-file FILE")
    pick = few_gold_sets if task.multi_label else few_gold
    # per text: a label index, or (multi-label) a list of them; fixed by --seed
    gold: list[Any] | None = pick(all_gold, a.gold_per_class, a.seed) if a.gold_per_class else None
    n_gold_file = 0
    if a.gold_file:  # labels a person added, e.g. to the texts `tinydistill suggest` picked
        from .data import read_examples

        given = {e.text: e.gold for e in read_examples(a.gold_file, task) if e.gold not in (None, [])}
        where = {t: i for i, t in enumerate(texts)}
        gold = list(gold) if gold else [None] * len(texts)
        for t, g in given.items():
            if t in where:
                gold[where[t]] = g
                n_gold_file += 1
        log(
            f"--gold-file: {n_gold_file} of {len(given)} labelled texts are in the training data and are used"
        )
    n_gold = sum(g is not None for g in gold) if gold else 0
    device = pick_device(a.device)
    scores = torch.tensor([r["scores"] for r in rows], dtype=torch.float64)
    probs = (scores.sigmoid() if task.multi_label else scores.softmax(-1)).float()  # the soft targets
    upper = [i for i, g in enumerate(all_gold) if g is not None]
    kgold = [i for i, g in enumerate(gold or []) if g is not None]
    for name, *_ in VARIANTS:  # drop models from an earlier train in this run directory
        shutil.rmtree(run / name, ignore_errors=True)
    shutil.rmtree(run / "seeds", ignore_errors=True)

    from .progress import Progress

    def batches(n: int) -> int:
        n -= round(a.val_share * n)  # held out from training
        return a.epochs * -(-n // a.batch_size)

    per_seed = batches(len(texts)) + (batches(len(kgold)) if a.gold_only else 0)
    per_seed += batches(len(upper)) if a.upper_bound else 0
    bar = Progress("train", a.seeds * per_seed, log, run / "status.json", unit="batches")
    t0 = time.perf_counter()
    meta = _meta(run)
    sel = {
        "val_share": a.val_share,
        "patience": a.patience,
        "chunk_tokens": meta.get("chunk_tokens"),
        "chunk_stride": meta.get("chunk_stride"),
        "init_head": [
            f"{v}: {d}" if d else v
            for v, d in zip(task.verbalisations, task.descriptions or [None] * len(task.labels), strict=True)
        ]
        if a.init_head
        else None,
        "outside": task.outside_id,
        # the tokens of the training texts only (test texts are new text, as in use, so test accuracy shows
        # what trimming costs); every variant and seed trims to the same vocabulary
        "trim_texts": texts if a.trim_vocab else None,
        "trim_keep": a.trim_keep,
    }
    if meta.get("chunk_tokens") and meta.get("chunk_tokenizer") != (encoder_of(a.student) or a.student):
        log(
            f"warning: the windows were cut with the tokenizer of {meta['chunk_tokenizer']}, not of "
            f"{a.student}; label again with --chunk-tokenizer {a.student} if they count tokens differently"
        )
    kept: dict[str, list[int]] = {}
    for i in range(a.seeds):
        seed = a.seed + i
        out = run if i == 0 else run / "seeds" / str(seed)
        hp = (a.epochs, a.batch_size, a.lr, a.max_length, device, seed, log, task.multi_label)
        log(f"seed {seed}: distilling {a.student} on {len(rows)} texts, {n_gold} with gold labels ({device})")
        kept.setdefault("student", []).append(
            train_student(
                texts,
                probs,
                task.labels,
                out / "student",
                a.student,
                gold,
                a.ce_weight,
                *hp,
                on_step=bar.update,
                **sel,
            )
        )
        if a.gold_only:
            log(f"seed {seed}: baseline, {a.student} on the {len(kgold)} gold labels only, no teacher")
            kept.setdefault("student-kgold", []).append(
                train_student(
                    [texts[j] for j in kgold],
                    None,
                    task.labels,
                    out / "student-kgold",
                    a.student,
                    [g for g in gold or [] if g is not None],  # the labels of kgold
                    0.0,
                    *hp,
                    on_step=bar.update,
                    **sel,
                )
            )
        if a.upper_bound:
            log(f"seed {seed}: upper bound, {a.student} on all {len(upper)} gold training labels, no teacher")
            kept.setdefault("student-gold", []).append(
                train_student(
                    [texts[j] for j in upper],
                    None,
                    task.labels,
                    out / "student-gold",
                    a.student,
                    [all_gold[j] for j in upper],
                    0.0,
                    *hp,
                    on_step=bar.update,
                    **sel,
                )
            )
    bar.finish()
    _meta(
        run,
        {
            "student": a.student,
            "epochs": a.epochs,
            "init_head": a.init_head,
            "val_share": a.val_share,
            "patience": a.patience if a.val_share else None,
            "kept_epochs": kept if a.val_share else None,
            "lr": a.lr,
            "batch_size": a.batch_size,
            "gold_per_class": a.gold_per_class,
            "gold_file": a.gold_file,
            "n_gold_file": n_gold_file,
            "n_gold": n_gold,
            "ce_weight": a.ce_weight if n_gold else 0.0,
            "student_device": device,
            "train_seconds": round(time.perf_counter() - t0, 1),
            "seeds": a.seeds,
            "gold_only": a.gold_only,
            "upper_bound": a.upper_bound,
            "n_gold_upper_bound": len(upper) if a.upper_bound else None,
            "trim_vocab": a.trim_vocab,
            "trim_keep": a.trim_keep if a.trim_vocab else None,
            "vocab_size": json.loads((run / "student" / "config.json").read_text(encoding="utf-8"))[
                "vocab_size"
            ],
        },
    )


def cmd_export(a: argparse.Namespace) -> None:
    from .export import export_onnx

    run = Path(a.run)
    for p in export_onnx(run / "student", run / "onnx", int8=a.int8):
        log(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")


def cmd_eval(a: argparse.Namespace) -> None:
    import torch

    from .evaluate import (
        accuracy,
        aggregate,
        bench,
        dir_size_mb,
        environment,
        hf_weights_mb,
        macro_f1,
        multilabel_metrics,
        ordinal_metrics,
        outside_metrics,
        render,
        time_calls,
        top_k_accuracy,
    )
    from .export import Predictor
    from .student import load_student, student_logits
    from .teacher import Ensemble, combine_scores

    run = Path(a.run)
    meta = _meta(run)
    task = load_task(run / "task.yaml")
    test = _read_jsonl(run / "test.jsonl")
    train = _read_jsonl(run / "train.jsonl")
    labelled = [r for r in test if r["gold"] is not None]
    gold = [r["gold"] for r in labelled]
    texts = [r["text"] for r in labelled]
    multi = task.multi_label

    def decide(scores) -> int | frozenset[int]:
        """A prediction from label scores or logits: the top label, or (multi-label) every positive one."""
        row = list(scores)
        return frozenset(i for i, x in enumerate(row) if x > 0) if multi else _argmax(row)

    if multi:
        gold = [frozenset(g) for g in gold]
    teacher_pred = [decide(r["scores"]) for r in labelled]
    bench_texts = [r["text"] for r in (test or train)]
    if not bench_texts:
        raise SystemExit("no texts to evaluate on: label a --test set (or training data) first")
    threads = torch.get_num_threads()

    def quality(pred: list, scores: list | None = None) -> dict:
        """Metrics of predictions; with each text's label scores, also top-3 accuracy (tasks with more than
        five labels)."""
        if not gold:
            return {}
        if multi:
            return multilabel_metrics(pred, gold, len(task.labels)) | {
                "agreement": accuracy(pred, teacher_pred)
            }
        extra = ordinal_metrics(pred, gold) if task.ordinal else {}
        if task.outside:
            extra |= outside_metrics(pred, gold, len(task.labels) - 1)
        if scores is not None and len(task.labels) > 5:
            extra["top3_accuracy"] = top_k_accuracy(scores, gold)
        return extra | {
            "accuracy": accuracy(pred, gold),
            "macro_f1": macro_f1(pred, gold),
            "agreement": accuracy(pred, teacher_pred),
        }

    models: list[dict] = []
    from .progress import Progress

    teachers = meta.get("teachers") or [meta["teacher"]]
    devices = [d for d in (a.teacher_devices or meta.get("teacher_device", "cpu")).split(",") if d]
    if meta.get("vllm_url"):
        devices = ["vllm"]  # the teacher runs on the server
    elif meta.get("api_url"):
        devices = ["api"]
    n_students = sum(len(_seed_dirs(run, name)) for name, *_ in VARIANTS)
    n_onnx = sum((run / "onnx" / f).exists() for f in ("model.onnx", "model.int8.onnx"))
    bar = Progress(
        "eval", len(teachers) * len(devices) + n_students + n_onnx, log, run / "status.json", "steps", every=0
    )
    n_prompts = 1 + len(meta.get("prompts") or [])  # each teacher's members: one per prompt
    prompt = (
        (f"{meta['few_shot']}-shot" if meta.get("few_shot") else "zero-shot")
        + (f", {n_prompts} prompts" if n_prompts > 1 else "")
        + (", 4-bit" if meta.get("load_4bit") else "")
        + (", reasoning" if meta.get("reason") else "")
    )
    if meta.get("few_shot_similar"):
        prompt = f"{meta['few_shot_similar']} similar examples" + prompt.partition("-shot")[2]
    examples = [(r["text"], r["gold"]) for r in _read_jsonl(run / "examples.jsonl")] or None
    members: list[dict] = []
    for ti, name in enumerate(teachers):
        own_scores = [
            combine_scores(r["teacher_scores"][ti * n_prompts : (ti + 1) * n_prompts], multi)
            if len(teachers) > 1
            else r["scores"]
            for r in labelled
        ]
        q = quality([decide(s) for s in own_scores], own_scores)
        q.pop("agreement", None)
        for dev in devices:
            label = f"teacher, {prompt}" + (f": {name.split('/')[-1]}" if len(teachers) > 1 else "")
            row = {
                "role": "teacher",
                "name": label,
                "device": dev,
                "model": name,
                **q,
                "size_mb": hf_weights_mb(name),
            }
            if not a.skip_teacher_bench:
                log(f"benchmarking the teacher {name} on {dev}")
                scorers = _run_members(run, meta, task, name, dev, examples)
                scorer = scorers[0]
                if hasattr(scorer, "model"):  # a local model (not a vLLM server)
                    row["params_m"] = sum(p.numel() for p in scorer.model.parameters()) / 1e6
                    row["size_mb"] = (
                        scorer.model.get_memory_footprint() / 1e6
                    )  # weights as loaded (e.g. 4-bit)
                # several texts are packed into shared forward passes; a prompt ensemble runs every prompt
                score = (Ensemble(scorers) if len(scorers) > 1 else scorer).score_many
                row.update(bench(score, bench_texts, a.teacher_reps))
                if row["latency_ms_b1"] * 32 * 4 < 180_000:  # batch 32 only when it takes under three minutes
                    row["latency_ms_b32"] = time_calls(score, bench_texts, 32, 3)
                    row["throughput_per_s"] = 32 / (row["latency_ms_b32"] / 1000)
                del scorer, scorers, score
                if dev.startswith("cuda"):
                    torch.cuda.empty_cache()
            members.append(row)
            bar.update()
    if len(teachers) == 1:
        models.extend(members)
    else:  # the ensemble is the actual teacher: it runs every member, so its latency is their sum
        models.extend(m | {"role": "teacher member"} for m in members)
        q = quality(teacher_pred, [r["scores"] for r in labelled])
        q.pop("agreement", None)
        for dev in devices:
            on_dev = [m for m in members if m["device"] == dev]
            row = {
                "role": "teacher",
                "name": f"teacher ensemble of {len(teachers)}, {prompt}",
                "device": dev,
                **q,
                "size_mb": sum(m.get("size_mb") or 0 for m in on_dev) or None,
            }
            for key in ("latency_ms_b1", "latency_ms_b32"):
                if all(m.get(key) for m in on_dev):
                    row[key] = sum(m[key] for m in on_dev)
            if row.get("latency_ms_b32"):
                row["throughput_per_s"] = 32 / (row["latency_ms_b32"] / 1000)
            elif row.get("latency_ms_b1"):
                row["throughput_per_s"] = 1000 / row["latency_ms_b1"]
            models.insert(len(models), row)

    for name, role, row_name in VARIANTS:
        dirs = _seed_dirs(run, name)
        if not dirs:
            continue
        if role == "baseline":
            row_name = f"baseline: student on {meta.get('gold_per_class')} gold labels per class, no teacher"
        log(f"evaluating {row_name} (PyTorch, CPU, {len(dirs)} seed{'s' if len(dirs) > 1 else ''})")
        per_seed = []
        for d in dirs:
            model, tok = load_student(d, "cpu")
            z = student_logits(model, tok, texts).tolist() if texts else []
            per_seed.append(quality([decide(x) for x in z], z))
            bar.update()
        row = {"role": role, "name": row_name, "device": "cpu", **aggregate(per_seed)}
        if role == "student":  # timings and size of the shipped (first-seed) model
            model, tok = load_student(dirs[0], "cpu")
            row["params_m"] = sum(p.numel() for p in model.parameters()) / 1e6
            row["size_mb"] = dir_size_mb(list(dirs[0].glob("*.safetensors")) + list(dirs[0].glob("*.bin")))
            row.update(
                bench(
                    lambda b, m=model, t=tok: student_logits(m, t, b, batch_size=32),
                    bench_texts,
                    a.reps,
                    max(3, a.reps // 5),
                )
            )
        models.append(row)
    for int8 in (False, True):
        f = run / "onnx" / ("model.int8.onnx" if int8 else "model.onnx")
        if not f.exists():
            continue
        log(f"evaluating {f.name} (onnxruntime, CPU)")
        pr = Predictor(run / "onnx", int8=int8, threads=threads)
        z = pr.logits(texts).tolist() if texts else []
        pred = [decide(x) for x in z]
        row = {
            "role": "student",
            "name": f"student, ONNX {'int8' if int8 else 'fp32'}",
            "device": "cpu",
            **quality(pred, z),
            "size_mb": dir_size_mb([f]),
        }
        # timed on real batches (batch 32 = one onnxruntime call with 32 texts)
        row.update(
            bench(lambda b, pr=pr: pr.logits(b, batch_size=len(b)), bench_texts, a.reps, max(3, a.reps // 5))
        )
        models.append(row)
        bar.update()

    order = {"teacher": 0, "teacher member": 1, "student": 2, "baseline": 3, "upper bound": 4}
    models.sort(key=lambda m: order[m["role"]])  # deployable rows first, reference rows last
    results: dict[str, Any] = {
        "task": task.name,
        "teacher": meta.get("teacher"),
        "student": meta.get("student"),
        "data": {
            "train": meta.get("data"),
            "test": meta.get("test"),
            "n_train": len(train),
            "n_test": len(labelled),
        },
        "training": {
            k: meta.get(k)
            for k in (
                "epochs",
                "val_share",
                "patience",
                "kept_epochs",
                "lr",
                "batch_size",
                "gold_per_class",
                "gold_file",
                "n_gold_file",
                "n_gold",
                "upper_bound",
                "gold_only",
                "seeds",
                "n_gold_upper_bound",
                "label_texts_scored",
                "few_shot",
                "few_shot_similar",
                "few_shot_model",
                "n_examples",
                "ce_weight",
                "student_device",
                "train_seconds",
                "teacher_device",
                "label_seconds",
                "load_4bit",
                "shortlist",
                "shortlist_model",
                "shortlist_keep",
                "chunk_tokens",
                "chunk_stride",
                "budget",
                "rounds",
            )
        },
        "environment": environment(threads),
        "models": models,
    }
    if meta.get("shortlist") and labelled and not multi:
        from .teacher import NOT_SCORED

        # the share of test texts whose gold label was among the K the teacher scored
        results["training"]["shortlist_recall"] = sum(
            r["scores"][r["gold"]] > NOT_SCORED / 2 for r in labelled
        ) / len(labelled)
    deployed = _deployed(run)
    if labelled and deployed is not None and not multi:  # the cascade is for single-label tasks
        from .evaluate import cascade_curve

        probs = deployed.predict_proba(texts)
        results["cascade"] = cascade_curve(
            probs.max(-1).tolist(), probs.argmax(-1).tolist(), teacher_pred, gold
        )
    (run / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    bar.finish()
    md = render(results)
    (run / "results.md").write_text(md, encoding="utf-8")
    print(md)


def cmd_report(a: argparse.Namespace) -> None:
    from .evaluate import render

    run = Path(a.run)
    md = render(json.loads((run / "results.json").read_text(encoding="utf-8")))
    (run / "results.md").write_text(md, encoding="utf-8")
    print(md)


def _deployed(run: Path):
    """The model a deployment would use: int8 ONNX if exported, else fp32 ONNX, else None."""
    from .export import Predictor

    onnx = run / "onnx"
    if (onnx / "model.int8.onnx").exists() or (onnx / "model.onnx").exists():
        return Predictor(onnx)
    return None


def _run_teacher(run: Path, device: str):
    """The run's teacher (or ensemble), with its prompt and few-shot examples, for scoring new texts."""
    from .teacher import Ensemble, Scorer

    meta, task = _meta(run), load_task(run / "task.yaml")
    examples = [(r["text"], r["gold"]) for r in _read_jsonl(run / "examples.jsonl")] or None
    members: list[Scorer] = [
        m
        for t in meta.get("teachers") or [meta["teacher"]]
        for m in _run_members(run, meta, task, t, device, examples)
    ]
    return members[0] if len(members) == 1 else Ensemble(members)


def cmd_predict(a: argparse.Namespace) -> None:
    from .export import Predictor

    texts = list(a.texts)
    if a.data:
        from .data import read_examples
        from .task import Task

        field = Task("", ["a", "b"], ["a", "b"], "{text}", text_field=a.text_field, label_field=None)
        texts += [e.text for e in read_examples(a.data, field)]
    if not texts:
        raise SystemExit("give texts as arguments or --data FILE")
    pr = Predictor(a.model, int8=False if a.fp32 else None)
    if a.cascade:
        from .export import Cascade

        cascade = Cascade(pr, _run_teacher(Path(a.cascade), a.device), a.threshold, a.log)
        for t, r in zip(texts, cascade.predict(texts), strict=True):
            r["confidence"] = round(r["confidence"], 4)
            print(json.dumps({"text": t, **r}, ensure_ascii=False))
        return
    for t, r in zip(texts, pr.describe(texts), strict=True):
        print(json.dumps({"text": t, **r}, ensure_ascii=False))


def cmd_serve(a: argparse.Namespace) -> None:
    from .export import Cascade, Predictor
    from .serve import make_server

    student = Predictor(a.model)
    model = (
        Cascade(student, _run_teacher(Path(a.cascade), a.device), a.threshold, a.log)
        if a.cascade
        else student
    )
    server = make_server(model, a.host, a.port)
    log(f"serving {a.model} on http://{a.host}:{server.server_port} (POST /predict, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def cmd_plot(a: argparse.Namespace) -> None:
    from .plots import save_all

    files = save_all(Path(a.run), a.out)
    if not files:
        raise SystemExit(f"{a.run} has no data to plot: run `tinydistill eval` first")
    for f in files:
        print(f)


def cmd_web(a: argparse.Namespace) -> None:
    import shutil

    from .explainer import build_explainer, render_explainer
    from .export import model_path
    from .hub import page_json
    from .web import build_site, clear_models, render_site

    hub = {}  # the other models for the picker: repo -> tinydistill-page.json
    for name in filter(None, (m.strip() for m in (a.models or "").split(","))):
        found = page_json(name.removeprefix("hf:"))
        if found is None:
            log(f"skipping {name}: it has no tinydistill-page.json (upload it again with `tinydistill push`)")
        else:
            hub[name.removeprefix("hf:")] = found
    # the explainer is the main page (out/index.html) and the plain page goes to out/simple/; a model with no
    # explainer (static, ModernBERT) gets the plain page as its only page
    out = Path(a.out)
    explainer: Path | None = None
    if a.remote:  # the pages' data from the repo's tinydistill-page.json; the model loads from the Hub
        if not a.model.startswith("hf:"):
            raise SystemExit("--remote needs a model on the Hub: hf:user/name")
        repo = a.model.removeprefix("hf:")
        data = page_json(repo)
        if data is None:
            raise SystemExit(f"{a.model} has no tinydistill-page.json: upload it again with `push`")
        for d in filter(None, (data["simple"], data.get("explainer"))):
            d["repo"] = repo
            d["text"]["title"] = a.title or d["text"]["title"]
        clear_models(out)  # an earlier build's model folders
        if data.get("explainer"):
            explainer = render_explainer(data["explainer"], out, a.note, hub)
        simple = out / "simple" if explainer else out
        clear_models(simple)
        page = render_site(data["simple"], simple, bool(explainer), a.note, hub)
    else:
        folder = model_path(a.model)
        config = folder / "config.json"
        if config.exists() and json.loads(config.read_text(encoding="utf-8")).get("tinydistill_chunk_tokens"):
            log(
                "warning: this model reads long texts in windows, but the page reads only the first one "
                "(predict, serve and Predictor read them all)"
            )
        try:
            explainer = build_explainer(folder, out, a.title, note=a.note, hub=hub)
        except ValueError as e:
            log(f"no explainer page: {e}")
        simple = out / "simple" if explainer else out
        page = build_site(folder, simple, a.title, note=a.note, explainer=bool(explainer), hub=hub)
    if explainer:
        old = out / "explainer"  # where the explainer was before: its links (with ?text=, ?tour=1) still work
        if old.is_dir():
            shutil.rmtree(old)
            old.mkdir()
            (old / "index.html").write_text(REDIRECT, encoding="utf-8")
        log(f"wrote {explainer}, the model's layers and heads as it reads a text, and {page}, a simpler page")
    else:
        log(f"wrote {page}")
    log(f"serve the folder over HTTP, e.g. python -m http.server -d {a.out}")


REDIRECT = """<!doctype html><meta charset="utf-8"><title>Moved</title>
<script>location.replace("../" + location.search + location.hash)</script>
<p>This page moved to <a href="../">the folder above</a>.</p>
"""


def cmd_synth(a: argparse.Namespace) -> None:
    import torch

    from .synth import NearDuplicates, generator, load_attributes, synthesise
    from .teacher import LabelScorer, load_teacher

    task = load_task(a.task)
    if a.data and a.dedup is None:
        raise SystemExit("--data is for --dedup: texts too similar to its texts are dropped")
    keep = None
    if a.dedup is not None:
        from .data import read_examples
        from .shortlist import Embedder

        existing = [e.text for e in read_examples(a.data, task, with_gold=False)] if a.data else []
        keep = NearDuplicates(Embedder().embed, a.dedup, existing)
    torch.manual_seed(a.seed)
    name = _teachers(a.teacher)[0]
    log(f"loading {name} to write example texts")
    scorer = load_teacher(name, task, a.device, a.load_4bit)  # its model and tokenizer write the texts
    if not isinstance(scorer, LabelScorer):
        raise SystemExit("synth needs a local teacher (not --vllm-url or --api-url)")
    n = synthesise(
        task,
        generator(scorer.model, scorer.tok, a.temperature),
        a.per_label,
        Path(a.out),
        log=log,
        attributes=load_attributes(a.task) if a.vary else None,
        seed=a.seed,
        keep=keep,
    )
    if keep:
        log(f"--dedup: dropped {keep.dropped} texts with a cosine similarity of {a.dedup} or more")
    log(f"wrote {n} texts to {a.out}; label them with `tinydistill run {a.task} --data {a.out} ...`")


def cmd_init(a: argparse.Namespace) -> None:
    from .task import task_yaml_from_data

    out = Path(a.out)
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists; use --force to overwrite it")
    out.write_text(task_yaml_from_data(a.data, a.text_field, a.label_field), encoding="utf-8")
    log(f"wrote {out}: edit the prompt and label names, then run `tinydistill run {out} --data ...`")


def cmd_push(a: argparse.Namespace) -> None:
    from .hub import push

    print(push(Path(a.model), a.repo, a.private, a.license, a.title, a.note))


def cmd_analyse(a: argparse.Namespace) -> None:
    from .analyse import build

    if a.review:
        import csv

        from .analyse import review

        task = load_task(Path(a.run) / "task.yaml")
        if task.multi_label:
            raise SystemExit("--review is for single-label tasks")
        queue, wrong = review(_read_jsonl(Path(a.run) / "train.jsonl"), task.labels)
        with Path(a.review).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, ["text", "gold", "teacher_label", "teacher_confidence", "member_labels"])
            w.writeheader()
            w.writerows(queue)
        log(
            f"wrote {len(queue)} training texts to {a.review}: {wrong} whose gold label the teacher "
            f"contradicts, then {len(queue) - wrong} on which the teacher's ensemble members disagree"
        )
        return
    page, summary = build(Path(a.run), a.top)
    (Path(a.run) / "analysis.html").write_text(page, encoding="utf-8")
    print(summary)
    log(f"wrote {a.run}/analysis.html")


def cmd_suggest(a: argparse.Namespace) -> None:
    import numpy as np

    from .export import Predictor
    from .suggest import pick, priority

    run = Path(a.run)
    task = load_task(run / "task.yaml")
    rows = _read_jsonl(run / "train.jsonl")
    used = {r["text"] for r in _read_jsonl(run / "examples.jsonl")}  # already shown to the teacher
    rows = [r for r in rows if r["text"] not in used]
    texts = [r["text"] for r in rows]
    scores = np.array([r["scores"] for r in rows], dtype=np.float64)
    if task.multi_label:
        p = 1 / (1 + np.exp(-scores))
    else:
        p = np.exp(scores - scores.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
    if not (run / "onnx").exists():
        raise SystemExit(f"{run}/onnx is missing: run `tinydistill export` first")
    q = Predictor(run / "onnx").predict_proba(texts)
    chosen = pick(priority(p, q, task.multi_label), p.argmax(-1), a.n)
    field = task.label_field or "label"
    with Path(a.out).open("w", encoding="utf-8") as f:
        for i in chosen:
            row = {
                task.text_field: texts[i],
                field: "",
                "teacher_label": task.labels[int(p[i].argmax())],
                "student_label": task.labels[int(q[i].argmax())],
                "teacher_confidence": round(float(p[i].max()), 3),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    disagree = sum(int(p[i].argmax() != q[i].argmax()) for i in chosen)
    log(
        f"wrote {len(chosen)} texts to {a.out} ({disagree} where student and teacher disagree). Fill in "
        f"'{field}' and train with --gold-file {a.out}"
    )


def cmd_tune(a: argparse.Namespace) -> None:
    from .data import read_examples
    from .teacher import pick_device
    from .tune import instructions, table, tune

    paths = [a.task, *a.prompt]
    if len({Path(p).name for p in paths}) < len(paths):  # the results name each prompt by its file name
        raise SystemExit("the task file and every --prompt file need different file names")
    task = load_task(a.task)
    tasks = [(Path(p).name, t) for p, t in zip(paths, [task, *_variants(task, a.prompt)], strict=True)]
    rows = list(read_examples(a.data, task, a.limit))
    texts, gold = [e.text for e in rows], [e.gold for e in rows]
    shots = [int(x) for x in a.few_shot.split(",")]
    results = tune(
        tasks,
        texts,
        gold,
        _teachers(a.teacher),
        shots,
        a.dev_size,
        pick_device(a.device),
        a.load_4bit,
        a.seed,
        log,
        a.propose,
        Path(a.out).name + "-",
        a.rounds,
    )
    n_dev = min(a.dev_size, sum(g not in (None, []) for g in gold))
    md = table(results, n_dev - n_dev // 2 if a.rounds > 1 else n_dev)
    if a.rounds > 1:
        md += f"The teacher saw mistakes on {n_dev // 2} other dev texts; those are not scored.\n"
    best = results[0]
    if a.propose:  # the candidates, and the best task file, next to OUT.md
        for name, t in tasks[len(paths) :]:
            _write_task(Path(a.out).with_name(name), t)
            paths.append(str(Path(a.out).with_name(name)))
        _write_task(Path(f"{a.out}-best.yaml"), dict(tasks)[best["prompt"]])
        md += instructions(results, tasks)
        log(f"wrote {len(tasks) - len(a.prompt) - 1} candidate task files and the best as {a.out}-best.yaml")
    task_path = next(p for p in paths if Path(p).name == best["prompt"])
    cmd = f"tinydistill run {task_path} --teacher {best['teacher']}" + (
        f" --few-shot {best['few_shot']}" if best["few_shot"] else ""
    )
    Path(f"{a.out}.md").write_text(md + f"\nBest: `{cmd} --data ... --out ...`\n", encoding="utf-8")
    Path(f"{a.out}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(md)
    log(f"best on the dev split: {cmd} --data {a.data} --out RUN")


def _new_run(src: Path, dst: Path, train: list[dict]) -> None:
    """A run directory with src's task, teacher settings, examples and test set, and these training rows."""
    import shutil

    dst.mkdir(parents=True, exist_ok=True)
    for name in (
        "task.yaml",
        "meta.json",
        "examples.jsonl",
        "test.jsonl",
        *(_meta(src).get("prompts") or []),
    ):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    with (dst / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_refresh(a: argparse.Namespace) -> None:
    src, dst = Path(a.run), Path(a.out)
    if dst.resolve() == src.resolve():
        raise SystemExit("--out must be a new run directory")
    logged = _read_jsonl(Path(a.log))
    if not logged:
        raise SystemExit(
            f"{a.log} has no logged texts: run predict or serve with --cascade RUN --log {a.log}"
        )
    rows = _read_jsonl(src / "train.jsonl")
    seen = {r["text"] for r in rows} | {r["text"] for r in _read_jsonl(src / "test.jsonl")}  # no test texts
    added = [
        {"text": r["text"], "gold": None, "scores": r["scores"]} for r in logged if r["text"] not in seen
    ]
    added = list({r["text"]: r for r in added}.values())  # each logged text once
    _new_run(src, dst, rows + added)
    _meta(dst, {"refreshed_from": str(src), "refresh_log": a.log, "refresh_added": len(added)})
    log(f"{dst}: {len(rows)} training texts from {src} plus {len(added)} the teacher answered in the cascade")
    # the new student trains as the run's did; a flag given with another value than its default wins
    defaults = vars(build_parser().parse_args(["refresh", str(src), "--log", "", "--out", ""]))
    for k, v in _run_settings(src).items():
        if getattr(a, k, None) == defaults.get(k):
            setattr(a, k, v)
    a.int8 = a.int8 or (src / "onnx" / "model.int8.onnx").exists()
    a.run = str(dst)
    cmd_train(a)
    cmd_export(a)
    cmd_eval(a)
    log(f"compare them with: tinydistill compare {src} {dst}")


# the training settings of a run (from its meta.json) that refresh and improve train the new student with
RUN_KEYS = (
    "student",
    "init_head",
    "epochs",
    "val_share",
    "patience",
    "lr",
    "batch_size",
    "gold_per_class",
    "gold_file",
    "ce_weight",
    "seeds",
    "trim_vocab",
    "trim_keep",
)


def _run_settings(src: Path) -> dict:
    """The training settings ``src`` was trained with: RUN_KEYS from its meta.json, and the student's
    max_length."""
    meta = _meta(src)
    settings = {k: meta[k] for k in RUN_KEYS if meta.get(k) is not None}
    config = src / "student" / "config.json"
    if config.exists():
        max_length = json.loads(config.read_text(encoding="utf-8")).get("tinydistill_max_length")
        settings |= {"max_length": max_length} if max_length else {}
    return settings


def cmd_improve(a: argparse.Namespace) -> None:
    import torch

    from .evaluate import accuracy
    from .improve import hold_out, jobs, table
    from .student import load_student, student_logits
    from .synth import generator, synthesise
    from .teacher import LabelScorer

    src = Path(a.run)
    out = Path(a.out or f"{src}-improve")
    task, meta = load_task(src / "task.yaml"), _meta(src)
    if task.multi_label:
        raise SystemExit("improve is for single-label tasks")
    if meta.get("vllm_url") or meta.get("api_url"):
        raise SystemExit("improve needs the run's teacher as a local model (not --vllm-url or --api-url)")
    if not 0 < a.held_out < 1:
        raise SystemExit("--held-out must be between 0 and 1")
    rows = _read_jsonl(src / "train.jsonl")
    if not rows:
        raise SystemExit(f"{src}/train.jsonl is missing or empty")
    held = hold_out(rows, a.held_out, a.seed)
    train = [r for i, r in enumerate(rows) if i not in held]
    val = [r for i, r in enumerate(rows) if i in held]
    val_texts = [r["text"] for r in val]
    target = [_argmax(r["scores"]) for r in val]  # the teacher's labels, what the student learns
    out.mkdir(parents=True, exist_ok=True)
    with (out / "held_out.jsonl").open("w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"{len(val)} of {len(rows)} training texts are held out to find confusions and compare rounds")
    # every train, export and eval default, then the run's training settings
    defaults = vars(build_parser().parse_args(["refresh", str(src), "--log", "", "--out", ""]))
    settings = _run_settings(src)
    known = {r["text"] for r in rows} | {r["text"] for r in _read_jsonl(src / "test.jsonl")}
    teacher: Any = None
    generate: Any = None
    pred: list[int] = []
    rounds: list[dict] = []
    for k in range(a.rounds + 1):
        d, added = out / f"round-{k}", 0
        if k:
            js = jobs(task, val_texts, pred, target)
            if not js:
                log("the student agrees with the teacher on every held-out text: nothing to target")
                break
            if teacher is None:
                # the teacher stays loaded while the students train
                teacher = _run_teacher(src, a.device)
                lm = getattr(teacher, "members", [teacher])[0]
                if not isinstance(lm, LabelScorer):
                    raise SystemExit("improve needs the run's teacher as a local model")
                generate = generator(lm.model, lm.tok, a.temperature)
            d.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(a.seed + k)
            per = max(1, a.per_round // len(js))
            log(f"round {k}: asking for {per} texts for each of {len(js)} confusions")
            synthesise(task, generate, per, d / "synth.jsonl", log=log, seed=a.seed + k, jobs=js)
            new = [r[task.text_field] for r in _read_jsonl(d / "synth.jsonl")]
            new = list(dict.fromkeys(t for t in new if t not in known))
            known |= set(new)
            scores = teacher.score_many(new) if new else []
            train = train + [{"text": t, "gold": None, "scores": s} for t, s in zip(new, scores, strict=True)]
            added = len(new)
        _new_run(src, d, train)
        _meta(d, {"improved_from": str(src), "improve_round": k, "improve_added": added})
        b = argparse.Namespace(
            **defaults
            | settings
            | {"run": str(d), "device": a.device, "seed": a.seed, "threads": a.threads}
            | {"int8": a.int8 or (src / "onnx" / "model.int8.onnx").exists(), "skip_teacher_bench": True}
        )
        cmd_train(b)
        cmd_export(b)
        cmd_eval(b)
        model, tok = load_student(d / "student", "cpu")
        pred = student_logits(model, tok, val_texts).argmax(-1).tolist()
        results = json.loads((d / "results.json").read_text(encoding="utf-8"))
        student = next(m for m in results["models"] if m["role"] == "student")
        rounds.append(
            {
                "round": k,
                "dir": str(d),
                "added": added,
                "train": len(train),
                "n_held_out": len(val),
                "held_out": accuracy(pred, target),
                "test": student.get("accuracy"),
            }
        )
        (out / "improve.json").write_text(json.dumps(rounds, indent=2), encoding="utf-8")
        (out / "improve.md").write_text(table(rounds), encoding="utf-8")
        log(f"round {k}: held-out agreement {rounds[-1]['held_out']:.3f}")
        if k and rounds[-1]["held_out"] <= max(r["held_out"] for r in rounds[:-1]):
            log("the held-out agreement did not improve: stopping")
            break
    print(table(rounds))


def cmd_status(a: argparse.Namespace) -> None:
    from .progress import describe_status

    print(describe_status(Path(a.run) / "status.json"))


def cmd_env(a: argparse.Namespace) -> None:
    import platform
    from importlib.metadata import PackageNotFoundError, version

    from . import __version__

    def ver(pkg: str) -> str:
        try:
            return version(pkg)
        except PackageNotFoundError:
            return "not installed"

    lines = {
        "tinydistill": __version__,
        "Python": platform.python_version(),
        "Platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "torch": ver("torch"),
    }
    try:
        import torch

        cuda = torch.cuda.is_available()
        lines["CUDA available"] = f"{cuda} (CUDA {torch.version.cuda})" if cuda else "False"
        if cuda:
            lines["GPU"] = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
    except ImportError:
        lines["CUDA available"] = "unknown (torch not installed)"
    for pkg in ("transformers", "onnxruntime", "datasets", "huggingface_hub", "matplotlib", "bitsandbytes"):
        lines[pkg] = ver(pkg)
    print("\n".join(f"- {k}: {v}" for k, v in lines.items()))


def cmd_compare(a: argparse.Namespace) -> None:
    from .evaluate import compare_table

    runs = []
    for r in a.runs:
        path = Path(r) / "results.json" if Path(r).is_dir() else Path(r)
        label = Path(r).name if Path(r).is_dir() else f"{path.parent.name}/{path.stem}"
        runs.append((label, json.loads(path.read_text(encoding="utf-8"))))
    print(compare_table(runs), end="")


def cmd_run(a: argparse.Namespace) -> None:
    import gc

    import torch

    a.run = a.out
    if a.budget:
        _budget_rounds(a)
        if a.dry_run:
            return
    else:
        cmd_label(a)
        if a.dry_run:
            return
        gc.collect()  # release the teacher before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cmd_train(a)
    cmd_export(a)
    cmd_eval(a)
    out = a.out
    log(
        f"\nDone. The model is in {out}/onnx and the results in {out}/results.md. Next:\n"
        f'  tinydistill predict {out}/onnx "some text"     classify texts\n'
        f"  tinydistill serve {out}/onnx                   a JSON endpoint on http://127.0.0.1:8000\n"
        f"  tinydistill web {out}/onnx --out site          a page that runs it in the browser"
    )


def _budget_rounds(a: argparse.Namespace) -> None:
    """run --budget N --rounds R: the teacher labels N training texts in R rounds, N/R each. Round 1 picks
    them at random (--seed); each later round picks the unlabelled texts the student just trained is least
    sure of (highest entropy). The student is trained after every round; the last is the final one."""
    import gc
    import random

    import torch

    from .data import read_examples
    from .student import load_student, student_logits
    from .suggest import uncertainty
    from .teacher import pick_device

    task = load_task(a.task)
    pool = list(dict.fromkeys(e.text for e in read_examples(a.data, task, a.limit, with_gold=False)))
    n = min(a.budget, len(pool))
    if not 1 <= a.rounds <= n:
        raise SystemExit(f"--rounds must be between 1 and the budget ({n} texts)")
    sizes = [n * (r + 1) // a.rounds - n * r // a.rounds for r in range(a.rounds)]
    rng = random.Random(a.seed)
    if a.dry_run:  # a random sample of N texts stands in for the ones the rounds would pick
        cmd_label(a, set(rng.sample(pool, n)))
        return
    chosen = set(rng.sample(pool, sizes[0]))
    for r, size in enumerate(sizes):
        if r:
            left = [t for t in pool if t not in chosen]
            model, tok = load_student(Path(a.out) / "student", pick_device(a.device))
            u = uncertainty(student_logits(model, tok, left), task.multi_label)
            chosen.update(left[i] for i in u.argsort(descending=True, stable=True)[:size].tolist())
            del model
        log(f"budget round {r + 1} of {a.rounds}: the teacher labels {len(chosen)} of {len(pool)} texts")
        cmd_label(a, chosen)
        gc.collect()  # release the teacher before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cmd_train(a)
    _meta(Path(a.out), {"rounds": a.rounds, "budget_pool": len(pool)})


# ---------- parser ----------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="auto", help="auto, cpu or cuda (teacher labelling and training)")
    common.add_argument("--threads", type=int, help="CPU threads for torch and onnxruntime")
    common.add_argument("--seed", type=int, default=0, help="seed for example selection and training")

    label = argparse.ArgumentParser(add_help=False)
    label.add_argument("task", help="task YAML file")
    label.add_argument(
        "--data", required=True, help="unlabelled texts: .csv, .jsonl or hf:<dataset>[:<config>]:<split>"
    )
    label.add_argument("--test", help="optional gold-labelled test set, same formats")
    label.add_argument("--out", required=True, help="run directory")
    label.add_argument(
        "--teacher",
        default=DEFAULT_TEACHER,
        help="Hugging Face causal LM; several, comma-separated, are ensembled",
    )
    label.add_argument(
        "--prompts",
        metavar="FILES",
        help="other task files (other prompts, same label ids in the same order), comma-separated: each "
        "teacher also scores every text with each of them, and the label probabilities are averaged "
        "(a prompt ensemble; local teachers only)",
    )
    label.add_argument(
        "--load-4bit", action="store_true", help="load the teacher in 4 bit (bitsandbytes, CUDA)"
    )
    label.add_argument(
        "--cache-dir",
        help="where teacher scores are cached (default: the run directory); "
        "runs that share it label each text once",
    )
    label.add_argument(
        "--vllm-url",
        help="score with this vLLM server (OpenAI API base, e.g. http://host:8000/v1), not a local model",
    )
    label.add_argument(
        "--api-url",
        help="score with a hosted OpenAI-compatible chat API (base URL, e.g. https://api.openai.com/v1; "
        "--teacher names the model there): the labels are shown as options A, B, C, ... and scored by "
        "the answer letter's log-probability; single-label tasks with at most 20 labels",
    )
    label.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        metavar="VAR",
        help="environment variable holding the API key for --api-url (default OPENAI_API_KEY; "
        "unset: no key is sent, as for a local server)",
    )
    label.add_argument(
        "--api-samples",
        type=int,
        metavar="N",
        help="for APIs without log-probabilities: sample N answers per text (N >= 2, N times the requests) "
        "and score each label by its share of the answers",
    )
    label.add_argument(
        "--api-temperature",
        type=float,
        default=0.7,
        metavar="T",
        help="sampling temperature for --api-samples (default 0.7)",
    )
    label.add_argument(
        "--shortlist",
        type=int,
        metavar="K",
        help="score only the K labels most similar to each text (for tasks with very many labels)",
    )
    label.add_argument("--shortlist-model", help="sentence-embedding model for --shortlist (default: MiniLM)")
    label.add_argument(
        "--shortlist-keep",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        metavar="IDS",
        help="label ids the teacher always scores besides the shortlist, comma-separated: catch-all labels "
        "such as other or out of scope, whose names resemble no particular text",
    )
    label.add_argument(
        "--few-shot",
        type=int,
        default=0,
        metavar="K",
        help="show K gold training examples per class in the teacher's prompt (needs {examples} in it)",
    )
    label.add_argument(
        "--few-shot-similar",
        type=int,
        default=0,
        metavar="M",
        help="show each text the M gold training examples most similar to it, most similar last (needs "
        "{examples} before {text}); with --few-shot K they are picked from those K per class",
    )
    label.add_argument(
        "--few-shot-model", help="sentence-embedding model for --few-shot-similar (default: MiniLM)"
    )
    label.add_argument(
        "--reason",
        action="store_true",
        help="the teacher thinks briefly before it answers (greedy, up to --reason-tokens), and the labels "
        "are scored after its reasoning; single-label tasks, local or vLLM teachers; much slower",
    )
    label.add_argument(
        "--reason-tokens",
        type=int,
        default=256,
        metavar="K",
        help="with --reason: the most tokens the teacher writes before its answer (default 256)",
    )
    label.add_argument(
        "--teacher-tokens",
        type=int,
        default=2048,
        help="tokens per teacher forward pass; short texts with few labels are packed together",
    )
    label.add_argument(
        "--chunk-tokens",
        type=int,
        metavar="T",
        help="cut texts longer than T tokens of the student's tokenizer into overlapping windows of T "
        "tokens: the teacher scores each window, the student trains on them and averages its windows' "
        "probabilities per text when it predicts (for documents longer than the student reads)",
    )
    label.add_argument(
        "--chunk-stride",
        type=int,
        metavar="S",
        help="with --chunk-tokens: tokens from one window's start to the next (default T // 2, so windows "
        "overlap by half; S = T for no overlap)",
    )
    label.add_argument(
        "--chunk-tokenizer",
        metavar="MODEL",
        help="with --chunk-tokens: the student whose tokenizer counts the tokens (default: the default "
        "student; `run` uses its --student); train warns if its --student differs",
    )
    label.add_argument("--limit", type=int, help="use only the first N training texts")
    label.add_argument("--test-limit", type=int, help="use only the first N test texts")
    label.add_argument(
        "--dry-run",
        action="store_true",
        help="with --api-url: print the requests and (approximate) tokens labelling would take, and with "
        "--price-in/--price-out its cost, for the texts not in the cache yet; scores nothing",
    )
    label.add_argument(
        "--price-in", type=float, metavar="P", help="with --dry-run: the API's price per million input tokens"
    )
    label.add_argument(
        "--price-out",
        type=float,
        metavar="Q",
        help="with --dry-run: the API's price per million output tokens",
    )

    train = argparse.ArgumentParser(add_help=False)
    train.add_argument(
        "--student",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face encoder, or `static` / `static:<encoder>` for a static-embedding student "
        "(no attention: much smaller and faster; try --epochs 20 --lr 1e-3); <encoder> may be a Model2Vec "
        "model such as minishlab/potion-base-8M",
    )
    train.add_argument(
        "--init-head",
        choices=["names"],
        help="`names`: before training, set each label's classifier weights to the student's own embedding "
        "of the label's name (and description), so the untrained student ranks labels by similarity to "
        "their names (default: a random head)",
    )
    train.add_argument("--epochs", type=int, default=10, help="passes over the training texts (default 10)")
    train.add_argument(
        "--val-share",
        type=float,
        default=0.0,
        metavar="S",
        help="hold out this share of the training texts and keep the epoch with the lowest loss on them, "
        "stopping early after --patience epochs without improvement (e.g. 0.05; default: off, keep the last)",
    )
    train.add_argument("--patience", type=int, default=3, help="epochs without improvement before stopping")
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--max-length", type=int, default=128)
    train.add_argument(
        "--gold-file",
        help="gold labels for some training texts (same text and label fields as the task), e.g. the texts "
        "`tinydistill suggest` picked, labelled by a person; used like --gold-per-class",
    )
    train.add_argument(
        "--gold-per-class",
        type=int,
        default=0,
        metavar="K",
        help="also use the gold labels of K training examples per class (cross-entropy term)",
    )
    train.add_argument("--ce-weight", type=float, default=1.0, help="weight of that cross-entropy term")
    train.add_argument(
        "--gold-only",
        action="store_true",
        help="also train the student on the K gold labels per class only, without the teacher (a baseline)",
    )
    train.add_argument(
        "--seeds", type=int, default=1, help="training seeds per student (mean and std in eval)"
    )
    train.add_argument(
        "--trim-vocab",
        action="store_true",
        help="before training, cut the student's vocabulary (its embedding matrix and tokenizer) to the "
        "tokens of the training and test texts, plus its special tokens and --trim-keep common ones; makes a "
        "multilingual student much smaller. A new text's word whose tokens were not kept is split into kept "
        "pieces, bytes or the unknown token",
    )
    train.add_argument(
        "--trim-keep",
        type=int,
        default=1000,  # student.TRIM_KEEP
        metavar="K",
        help="with --trim-vocab: also keep the K most frequent tokens of the original vocabulary, for texts "
        "with tokens the training texts lack (default 1000)",
    )
    train.add_argument(
        "--upper-bound",
        action="store_true",
        help="also train the student on all gold training labels, without the teacher",
    )

    export = argparse.ArgumentParser(add_help=False)
    export.add_argument("--int8", action="store_true", help="also write an int8 dynamically quantised model")

    ev = argparse.ArgumentParser(add_help=False)
    ev.add_argument(
        "--teacher-devices",
        help="comma-separated devices to time the teacher on (default: the labelling device)",
    )
    ev.add_argument("--skip-teacher-bench", action="store_true", help="report teacher quality only")
    ev.add_argument("--reps", type=int, default=50, help="timed runs per student benchmark at batch 1")
    ev.add_argument("--teacher-reps", type=int, default=10, help="timed runs of the teacher (batch 1)")

    run_dir = argparse.ArgumentParser(add_help=False)
    run_dir.add_argument("run", help="run directory")

    p = argparse.ArgumentParser(
        prog="tinydistill", description="Distil an LLM classification prompt into a small, fast model."
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("label", parents=[label, common], help="score every label with the teacher").set_defaults(
        fn=cmd_label
    )
    sub.add_parser("train", parents=[run_dir, train, common], help="distil the student").set_defaults(
        fn=cmd_train
    )
    sub.add_parser(
        "export", parents=[run_dir, export, common], help="export the student to ONNX"
    ).set_defaults(fn=cmd_export)
    sub.add_parser("eval", parents=[run_dir, ev, common], help="quality and speed table").set_defaults(
        fn=cmd_eval
    )
    pr = sub.add_parser("predict", parents=[common], help="classify texts with an exported model")
    pr.add_argument("model", help="directory written by `export` (run/onnx), or hf:user/name on the Hub")
    pr.add_argument("texts", nargs="*")
    pr.add_argument("--data", help="file with texts (.csv, .jsonl or hf:...)")
    pr.add_argument("--text-field", default="text")
    pr.add_argument("--fp32", action="store_true", help="use the fp32 model even if an int8 one exists")
    pr.add_argument(
        "--cascade", metavar="RUN", help="send texts the student is unsure about to RUN's teacher"
    )
    pr.add_argument("--log", help="with --cascade: append the texts the teacher answered (for `refresh`)")
    pr.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="with --cascade: the student's top probability below which the teacher answers",
    )
    pr.set_defaults(fn=cmd_predict)
    srv = sub.add_parser("serve", parents=[common], help="serve an exported model over HTTP (JSON)")
    srv.add_argument("model", help="directory written by `export` (run/onnx), or hf:user/name on the Hub")
    srv.add_argument(
        "--host", default="127.0.0.1", help="interface to listen on (default: this machine only)"
    )
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument(
        "--cascade", metavar="RUN", help="send texts the student is unsure about to RUN's teacher"
    )
    srv.add_argument("--threshold", type=float, default=0.5, help="with --cascade: see `predict --help`")
    srv.add_argument("--log", help="with --cascade: append the texts the teacher answered (for `refresh`)")
    srv.set_defaults(fn=cmd_serve)
    plo = sub.add_parser(
        "plot",
        help="charts of a run as PNG files: training, speed and accuracy, per label, confusions, cascade",
    )
    plo.add_argument("run", help="run directory")
    plo.add_argument("--out", help="folder for the PNG files (default: RUN/plots)")
    plo.set_defaults(fn=cmd_plot)
    web = sub.add_parser("web", help="a static web page that runs the model in the browser")
    web.add_argument("model", help="directory written by `export` (run/onnx), or hf:user/name on the Hub")
    web.add_argument("--out", default="site", help="folder to write the page and model into")
    web.add_argument("--title", help="page title (default: from the task name)")
    web.add_argument("--note", help="an extra footer line, e.g. the dataset's licence and source")
    web.add_argument(
        "--models",
        help="other models for a menu in the pages, comma-separated: hf:user/a,hf:user/b (uploaded with "
        "`push`); visitors pick one and it loads from the Hub",
    )
    web.add_argument(
        "--remote",
        action="store_true",
        help="for MODEL=hf:user/name: write the pages with no model files; the model loads from the Hub",
    )
    web.set_defaults(fn=cmd_web)
    syn = sub.add_parser("synth", parents=[common], help="have the teacher write unlabelled example texts")
    syn.add_argument("task", help="task YAML file")
    syn.add_argument("--per-label", type=int, required=True, help="texts to write per label")
    syn.add_argument("--out", required=True, help="JSONL file to write (usable as --data)")
    syn.add_argument(
        "--teacher", default=DEFAULT_TEACHER, help="the model that writes (the first if several)"
    )
    syn.add_argument("--temperature", type=float, default=0.9)
    syn.add_argument("--load-4bit", action="store_true", help="load the model in 4 bit (bitsandbytes, CUDA)")
    syn.add_argument(
        "--vary",
        action="store_true",
        help="each request also names a length, a tone and a writer's situation, sampled with --seed (the "
        "task file's `attributes:` mapping replaces these)",
    )
    syn.add_argument(
        "--dedup",
        type=float,
        metavar="T",
        help="drop texts whose cosine similarity (sentence embeddings, MiniLM) to a text kept earlier, or to "
        "one in --data, is T or more (e.g. 0.95)",
    )
    syn.add_argument(
        "--data", help="with --dedup: existing texts (e.g. the real data) the new ones must differ from"
    )
    syn.set_defaults(fn=cmd_synth)
    ini = sub.add_parser("init", help="write a starter task file from a dataset's labels")
    ini.add_argument("data", help=".csv, .jsonl or hf:<dataset>[:<config>]:<split> with gold labels")
    ini.add_argument("--out", default="task.yaml")
    ini.add_argument("--text-field", default="text")
    ini.add_argument("--label-field", default="label")
    ini.add_argument("--force", action="store_true", help="overwrite an existing file")
    ini.set_defaults(fn=cmd_init)
    psh = sub.add_parser("push", help="upload an exported model to the Hugging Face Hub, with a model card")
    psh.add_argument("model", help="directory written by `export` (run/onnx)")
    psh.add_argument("--repo", required=True, help="Hub repository, e.g. user/my-classifier")
    psh.add_argument("--private", action="store_true")
    psh.add_argument(
        "--license", help="licence identifier for the model card (check student, teacher and data)"
    )
    psh.add_argument("--title", help="the model's title in the browser pages (default: from the task name)")
    psh.add_argument("--note", help="the pages' footer line for this model, e.g. the dataset and its licence")
    psh.set_defaults(fn=cmd_push)
    sub.add_parser(
        "report", parents=[run_dir], help="rebuild RUN/results.md from RUN/results.json"
    ).set_defaults(fn=cmd_report)
    tun = sub.add_parser("tune", parents=[common], help="choose teacher, prompt and few-shot on a dev split")
    tun.add_argument("task", help="task YAML file")
    tun.add_argument("--data", required=True, help="training data with gold labels for some texts")
    tun.add_argument("--teacher", default=DEFAULT_TEACHER, help="one or more teachers, comma-separated")
    tun.add_argument("--few-shot", default="0,8", help="comma-separated values of K to try (0 = zero-shot)")
    tun.add_argument("--prompt", action="append", default=[], help="another task file to try (same labels)")
    tun.add_argument(
        "--propose",
        type=int,
        default=0,
        metavar="N",
        help="the (first) teacher writes N rewrites of the task's instruction, the text before its first "
        "placeholder (the rest of the prompt is kept as written); they are tried too, as is the task "
        "without label descriptions. Writes each candidate as OUT-<name>.yaml and the best task file as "
        "OUT-best.yaml",
    )
    tun.add_argument(
        "--rounds",
        type=int,
        default=1,
        metavar="R",
        help="with --propose: up to R rounds. Each later round shows the teacher the best instruction's "
        "mistakes on one half of the dev split and asks for N rewrites that fix them (OUT-r2-1.yaml, ...); "
        "every candidate is scored on the other half. Stops at the first round without an improvement",
    )
    tun.add_argument("--dev-size", type=int, default=200, help="gold-labelled training texts to score")
    tun.add_argument("--limit", type=int, help="use only the first N training texts")
    tun.add_argument("--load-4bit", action="store_true", help="load teachers in 4 bit (bitsandbytes, CUDA)")
    tun.add_argument("--out", default="tune", help="writes OUT.md and OUT.json")
    tun.set_defaults(fn=cmd_tune)
    ref = sub.add_parser(
        "refresh",
        parents=[run_dir, train, export, ev, common],
        help="retrain RUN with the texts its cascade sent to the teacher (a new run directory)",
    )
    ref.add_argument("--log", required=True, help="the log written by predict/serve --cascade RUN --log FILE")
    ref.add_argument("--out", required=True, help="new run directory")
    ref.set_defaults(fn=cmd_refresh)
    imp = sub.add_parser(
        "improve",
        parents=[run_dir, common],
        help="rounds of teacher-written hard examples for the labels the student confuses (new runs)",
    )
    imp.add_argument("--rounds", type=int, default=3, help="rounds at most (default 3)")
    imp.add_argument("--per-round", type=int, default=200, metavar="N", help="new texts asked for per round")
    imp.add_argument(
        "--held-out",
        type=float,
        default=0.1,
        metavar="S",
        help="share of the training texts held out to find the confusions and compare rounds, against the "
        "teacher's labels (default 0.1; never the test set)",
    )
    imp.add_argument("--temperature", type=float, default=0.9, help="sampling temperature of the teacher")
    imp.add_argument(
        "--int8", action="store_true", help="also export int8 models (as the run did if it has one)"
    )
    imp.add_argument("--out", help="folder for the rounds (default: RUN-improve)")
    imp.set_defaults(fn=cmd_improve)
    sug = sub.add_parser("suggest", parents=[run_dir], help="which training texts a person should label next")
    sug.add_argument("--n", type=int, default=100, help="how many texts")
    sug.add_argument("--out", default="to_label.jsonl", help="JSONL to write, with an empty label to fill in")
    sug.set_defaults(fn=cmd_suggest)
    ana = sub.add_parser(
        "analyse", parents=[run_dir], help="where the teacher and student go wrong (RUN/analysis.html)"
    )
    ana.add_argument("--top", type=int, default=15, help="how many confusions and disagreements to list")
    ana.add_argument(
        "--review",
        metavar="OUT.csv",
        help="instead of the page, write a CSV of training texts for a person to check: where the teacher's "
        "label contradicts the gold label, the teacher's most confident first, then (ensembles) where its "
        "members disagree most. Training is not changed",
    )
    ana.set_defaults(fn=cmd_analyse)
    sub.add_parser(
        "status", parents=[run_dir], help="how far a run is, and whether it is still running"
    ).set_defaults(fn=cmd_status)
    cmp = sub.add_parser("compare", help="one markdown table comparing several runs (e.g. student sizes)")
    cmp.add_argument("runs", nargs="+", help="run directories or results.json files")
    cmp.set_defaults(fn=cmd_compare)
    sub.add_parser("env", help="versions and hardware, for bug reports").set_defaults(fn=cmd_env)
    rn = sub.add_parser(
        "run", parents=[label, train, export, ev, common], help="label, train, export and eval"
    )
    rn.add_argument(
        "--budget",
        type=int,
        metavar="N",
        help="the teacher labels only N training texts (for paid APIs): N/R at random, then in each of "
        "--rounds R rounds the N/R unlabelled texts the student trained so far is least sure of",
    )
    rn.add_argument("--rounds", type=int, default=4, metavar="R", help="with --budget: rounds (default 4)")
    rn.set_defaults(fn=cmd_run)
    return p


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    if getattr(a, "threads", None):
        import torch

        torch.set_num_threads(a.threads)
    a.fn(a)


if __name__ == "__main__":
    main()
