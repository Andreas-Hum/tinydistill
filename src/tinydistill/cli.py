"""Command-line interface: label -> train -> export -> eval, or `run` for all of them."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from .task import load_task

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
    meta = json.loads(path.read_text()) if path.exists() else {}
    if update:
        meta.update(update)
        path.write_text(json.dumps(meta, indent=2))
    return meta


def _argmax(xs) -> int:
    return max(range(len(xs)), key=xs.__getitem__)


# ---------- steps ----------


def _teachers(spec: str) -> list[str]:
    return [t.strip() for t in spec.split(",") if t.strip()]


def cmd_label(a: argparse.Namespace) -> None:
    import gc

    import torch

    from .data import count_rows, read_examples
    from .progress import Progress
    from .teacher import ScoreCache, combine_scores, iter_unique, load_teacher

    task = load_task(a.task)
    run = Path(a.out)
    run.mkdir(parents=True, exist_ok=True)
    (run / "task.yaml").write_text(
        yaml.safe_dump(task.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    examples = None
    if a.few_shot and "{examples}" not in task.prompt:
        raise SystemExit("--few-shot needs an {examples} line in the task's prompt")
    if a.few_shot:
        from .student import few_shot_examples

        rows = list(read_examples(a.data, task, a.limit))
        if not any(e.gold is not None for e in rows):
            raise SystemExit("--few-shot needs gold labels in the training data")
        examples = few_shot_examples([e.text for e in rows], [e.gold for e in rows], a.few_shot, a.seed)
        with (run / "examples.jsonl").open("w", encoding="utf-8") as f:
            for text, y in examples:
                f.write(json.dumps({"text": text, "gold": y}) + "\n")
        log(f"few-shot prompt with {len(examples)} gold examples ({a.few_shot} per class)")
    cache_dir = Path(a.cache_dir) if a.cache_dir else run
    cache_dir.mkdir(parents=True, exist_ok=True)
    splits = [
        (n, spec, lim)
        for n, spec, lim in (("train", a.data, a.limit), ("test", a.test, a.test_limit))
        if spec
    ]

    caches, fingerprints, device = [], [], "cpu"
    seconds, scored = 0.0, 0
    for name in _teachers(a.teacher):  # one teacher in memory at a time
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
        )
        fingerprints.append(scorer.fingerprint())
        device = str(scorer.device)
        cache = ScoreCache(cache_dir / f"teacher-{fingerprints[-1]}.jsonl")
        if cache.scores:
            log(f"resuming: {len(cache.scores)} texts already scored")
        for split, spec, limit in splits:
            t0 = time.perf_counter()
            bar = Progress(
                f"label {split} ({name.split('/')[-1]})", count_rows(spec, limit), log, run / "status.json"
            )
            texts = iter_unique(e.text for e in read_examples(spec, task, limit, with_gold=False))
            n = cache.fill(scorer, texts, on_read=bar.update)
            bar.finish()
            seconds += time.perf_counter() - t0
            scored += n
            log(f"{split}: {n} texts scored now by {name}, the rest from the cache")
        caches.append(cache)
        del scorer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for split, spec, limit in splits:
        rows = 0
        with (run / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for e in read_examples(spec, task, limit):
                per = [c.get(e.text) for c in caches]
                row = {"text": e.text, "gold": e.gold, "scores": combine_scores(per, task.multi_label)}
                if len(per) > 1:
                    row["teacher_scores"] = per
                f.write(json.dumps(row) + "\n")
                rows += 1
        log(f"{split}: {rows} texts")
    _meta(
        run,
        {
            "teacher": a.teacher,
            "teachers": _teachers(a.teacher),
            "few_shot": a.few_shot,
            "n_examples": len(examples or []),
            "load_4bit": a.load_4bit,
            "teacher_device": device,
            "fingerprint": fingerprints[0],
            "fingerprints": fingerprints,
            "data": a.data,
            "test": a.test,
            "label_seconds": round(seconds, 1),
            "label_texts_scored": scored,
            "teacher_tokens": a.teacher_tokens,
            "vllm_url": a.vllm_url,
            "shortlist": a.shortlist,
            "shortlist_model": a.shortlist_model,
        },
    )


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
    gold = pick(all_gold, a.gold_per_class, a.seed) if a.gold_per_class else None  # fixed by --seed
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
        return a.epochs * -(-n // a.batch_size)

    per_seed = batches(len(texts)) + (batches(len(kgold)) if a.gold_only else 0)
    per_seed += batches(len(upper)) if a.upper_bound else 0
    bar = Progress("train", a.seeds * per_seed, log, run / "status.json", unit="batches")
    t0 = time.perf_counter()
    for i in range(a.seeds):
        seed = a.seed + i
        out = run if i == 0 else run / "seeds" / str(seed)
        hp = (a.epochs, a.batch_size, a.lr, a.max_length, device, seed, log, task.multi_label)
        log(f"seed {seed}: distilling {a.student} on {len(rows)} texts, {n_gold} with gold labels ({device})")
        train_student(
            texts, probs, task.labels, out / "student", a.student, gold, a.ce_weight, *hp, on_step=bar.update
        )
        if a.gold_only:
            log(f"seed {seed}: baseline, {a.student} on the {len(kgold)} gold labels only, no teacher")
            train_student(
                [texts[j] for j in kgold],
                None,
                task.labels,
                out / "student-kgold",
                a.student,
                [gold[j] for j in kgold],
                0.0,
                *hp,
                on_step=bar.update,
            )
        if a.upper_bound:
            log(f"seed {seed}: upper bound, {a.student} on all {len(upper)} gold training labels, no teacher")
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
            )
    bar.finish()
    _meta(
        run,
        {
            "student": a.student,
            "epochs": a.epochs,
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
        render,
        time_calls,
    )
    from .export import Predictor
    from .student import load_student, student_logits
    from .teacher import load_teacher

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

    def quality(pred: list) -> dict:
        if not gold:
            return {}
        if multi:
            return multilabel_metrics(pred, gold, len(task.labels)) | {
                "agreement": accuracy(pred, teacher_pred)
            }
        extra = ordinal_metrics(pred, gold) if task.ordinal else {}
        return extra | {
            "accuracy": accuracy(pred, gold),
            "macro_f1": macro_f1(pred, gold),
            "agreement": accuracy(pred, teacher_pred),
        }

    models: list[dict] = []
    from .progress import Progress

    n_teacher = 0 if a.skip_teacher_bench else len(meta.get("teachers") or [meta["teacher"]])
    n_devices = 1 if meta.get("vllm_url") else len([d for d in (a.teacher_devices or "x").split(",") if d])
    n_students = sum(len(_seed_dirs(run, name)) for name, *_ in VARIANTS)
    n_onnx = sum((run / "onnx" / f).exists() for f in ("model.onnx", "model.int8.onnx"))
    bar = Progress(
        "eval", n_teacher * n_devices + n_students + n_onnx, log, run / "status.json", "steps", every=0
    )
    teachers = meta.get("teachers") or [meta["teacher"]]
    prompt = (f"{meta['few_shot']}-shot" if meta.get("few_shot") else "zero-shot") + (
        ", 4-bit" if meta.get("load_4bit") else ""
    )
    examples = [(r["text"], r["gold"]) for r in _read_jsonl(run / "examples.jsonl")] or None
    devices = [d for d in (a.teacher_devices or meta.get("teacher_device", "cpu")).split(",") if d]
    if meta.get("vllm_url"):
        devices = ["vllm"]  # the teacher runs on the server
    members: list[dict] = []
    for ti, name in enumerate(teachers):
        own = [decide(r["teacher_scores"][ti]) for r in labelled] if len(teachers) > 1 else teacher_pred
        q = quality(own)
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
                scorer = load_teacher(
                    name,
                    task,
                    dev,
                    meta.get("load_4bit", False),
                    meta.get("teacher_tokens", 2048),
                    examples,
                    meta.get("vllm_url"),
                    meta.get("shortlist"),
                    meta.get("shortlist_model"),
                )
                if hasattr(scorer, "model"):  # a local model (not a vLLM server)
                    row["params_m"] = sum(p.numel() for p in scorer.model.parameters()) / 1e6
                    row["size_mb"] = (
                        scorer.model.get_memory_footprint() / 1e6
                    )  # weights as loaded (e.g. 4-bit)
                score = scorer.score_many  # several texts are packed into shared forward passes
                row.update(bench(score, bench_texts, a.teacher_reps))
                if row["latency_ms_b1"] * 32 * 4 < 180_000:  # batch 32 only when it takes under three minutes
                    row["latency_ms_b32"] = time_calls(score, bench_texts, 32, 3)
                    row["throughput_per_s"] = 32 / (row["latency_ms_b32"] / 1000)
                del scorer
                if dev.startswith("cuda"):
                    torch.cuda.empty_cache()
            members.append(row)
            bar.update()
    if len(teachers) == 1:
        models.extend(members)
    else:  # the ensemble is the actual teacher: it runs every member, so its latency is their sum
        models.extend(m | {"role": "teacher member"} for m in members)
        q = quality(teacher_pred)
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
            per_seed.append(
                quality([decide(z) for z in student_logits(model, tok, texts).tolist()] if texts else [])
            )
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
        pred = [decide(z) for z in pr.logits(texts).tolist()] if texts else []
        row = {
            "role": "student",
            "name": f"student, ONNX {'int8' if int8 else 'fp32'}",
            "device": "cpu",
            **quality(pred),
            "size_mb": dir_size_mb([f]),
        }
        row.update(bench(lambda b, pr=pr: pr.logits(b), bench_texts, a.reps, max(3, a.reps // 5)))
        models.append(row)
        bar.update()

    order = {"teacher": 0, "teacher member": 1, "student": 2, "baseline": 3, "upper bound": 4}
    models.sort(key=lambda m: order[m["role"]])  # deployable rows first, reference rows last
    results = {
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
                "n_examples",
                "ce_weight",
                "student_device",
                "train_seconds",
                "teacher_device",
                "label_seconds",
                "load_4bit",
                "shortlist",
                "shortlist_model",
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
    (run / "results.json").write_text(json.dumps(results, indent=2))
    bar.finish()
    md = render(results)
    (run / "results.md").write_text(md)
    print(md)


def cmd_report(a: argparse.Namespace) -> None:
    from .evaluate import render

    run = Path(a.run)
    md = render(json.loads((run / "results.json").read_text()))
    (run / "results.md").write_text(md)
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
    from .teacher import Ensemble, load_teacher

    meta, task = _meta(run), load_task(run / "task.yaml")
    examples = [(r["text"], r["gold"]) for r in _read_jsonl(run / "examples.jsonl")] or None
    members = [
        load_teacher(
            t,
            task,
            device,
            meta.get("load_4bit", False),
            meta.get("teacher_tokens", 2048),
            examples,
            meta.get("vllm_url"),
            meta.get("shortlist"),
            meta.get("shortlist_model"),
        )
        for t in meta.get("teachers") or [meta["teacher"]]
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

    model = Predictor(a.model)
    if a.cascade:
        model = Cascade(model, _run_teacher(Path(a.cascade), a.device), a.threshold, a.log)
    server = make_server(model, a.host, a.port)
    log(f"serving {a.model} on http://{a.host}:{server.server_port} (POST /predict, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def cmd_web(a: argparse.Namespace) -> None:
    from .web import build_site

    page = build_site(Path(a.model), Path(a.out), a.title, note=a.note)
    log(f"wrote {page}; serve the folder over HTTP, e.g. python -m http.server -d {a.out}")


def cmd_synth(a: argparse.Namespace) -> None:
    import torch

    from .synth import generator, synthesise
    from .teacher import load_teacher

    task = load_task(a.task)
    torch.manual_seed(a.seed)
    name = _teachers(a.teacher)[0]
    log(f"loading {name} to write example texts")
    scorer = load_teacher(name, task, a.device, a.load_4bit)  # its model and tokenizer write the texts
    n = synthesise(
        task, generator(scorer.model, scorer.tok, a.temperature), a.per_label, Path(a.out), log=log
    )
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

    print(push(Path(a.model), a.repo, a.private, a.license))


def cmd_analyse(a: argparse.Namespace) -> None:
    from .analyse import build

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
    from .tune import table, tune

    paths = [a.task, *a.prompt]
    tasks = [(Path(p).name, load_task(p)) for p in paths]
    if len({tuple(t.labels) for _, t in tasks}) > 1:
        raise SystemExit("every --prompt file must have the same labels as the task")
    rows = list(read_examples(a.data, tasks[0][1], a.limit))
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
    )
    n_dev = min(a.dev_size, sum(g not in (None, []) for g in gold))
    md = table(results, n_dev)
    best = results[0]
    task_path = next(p for p in paths if Path(p).name == best["prompt"])
    cmd = f"tinydistill run {task_path} --teacher {best['teacher']}" + (
        f" --few-shot {best['few_shot']}" if best["few_shot"] else ""
    )
    Path(a.out).with_suffix(".md").write_text(md + f"\nBest: `{cmd} --data ... --out ...`\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps(results, indent=2))
    print(md)
    log(f"best on the dev split: {cmd} --data {a.data} --out RUN")


def cmd_refresh(a: argparse.Namespace) -> None:
    import shutil

    src, dst = Path(a.run), Path(a.out)
    if dst.resolve() == src.resolve():
        raise SystemExit("--out must be a new run directory")
    logged = _read_jsonl(Path(a.log))
    if not logged:
        raise SystemExit(
            f"{a.log} has no logged texts: run predict or serve with --cascade RUN --log {a.log}"
        )
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("task.yaml", "meta.json", "examples.jsonl", "test.jsonl"):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    rows = _read_jsonl(src / "train.jsonl")
    seen = {r["text"] for r in rows}
    added = [
        {"text": r["text"], "gold": None, "scores": r["scores"]} for r in logged if r["text"] not in seen
    ]
    added = list({r["text"]: r for r in added}.values())  # each logged text once
    with (dst / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in rows + added:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _meta(dst, {"refreshed_from": str(src), "refresh_log": a.log, "refresh_added": len(added)})
    log(f"{dst}: {len(rows)} training texts from {src} plus {len(added)} the teacher answered in the cascade")
    a.run = str(dst)
    cmd_train(a)
    cmd_export(a)
    cmd_eval(a)
    log(f"compare them with: tinydistill compare {src} {dst}")


def cmd_status(a: argparse.Namespace) -> None:
    from .progress import describe_status

    print(describe_status(Path(a.run) / "status.json"))


def cmd_compare(a: argparse.Namespace) -> None:
    from .evaluate import compare_table

    runs = []
    for r in a.runs:
        path = Path(r) / "results.json" if Path(r).is_dir() else Path(r)
        label = Path(r).name if Path(r).is_dir() else f"{path.parent.name}/{path.stem}"
        runs.append((label, json.loads(path.read_text())))
    print(compare_table(runs), end="")


def cmd_run(a: argparse.Namespace) -> None:
    import gc

    import torch

    a.run = a.out
    cmd_label(a)
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
        "--shortlist",
        type=int,
        metavar="K",
        help="score only the K labels most similar to each text (for tasks with very many labels)",
    )
    label.add_argument("--shortlist-model", help="sentence-embedding model for --shortlist (default: MiniLM)")
    label.add_argument(
        "--few-shot",
        type=int,
        default=0,
        metavar="K",
        help="show K gold training examples per class in the teacher's prompt (needs {examples} in it)",
    )
    label.add_argument(
        "--teacher-tokens",
        type=int,
        default=2048,
        help="tokens per teacher forward pass; short texts with few labels are packed together",
    )
    label.add_argument("--limit", type=int, help="use only the first N training texts")
    label.add_argument("--test-limit", type=int, help="use only the first N test texts")

    train = argparse.ArgumentParser(add_help=False)
    train.add_argument(
        "--student",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face encoder, or `static` / `static:<encoder>` for a static-embedding student "
        "(no attention: much smaller and faster, usually less accurate; try --lr 1e-3)",
    )
    train.add_argument("--epochs", type=int, default=5)
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
    pr.add_argument("model", help="directory written by `export` (run/onnx)")
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
    srv.add_argument("model", help="directory written by `export` (run/onnx)")
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
    web = sub.add_parser("web", help="a static web page that runs the model in the browser")
    web.add_argument("model", help="directory written by `export` (run/onnx)")
    web.add_argument("--out", default="site", help="folder to write the page and model into")
    web.add_argument("--title", help="page title (default: from the task name)")
    web.add_argument("--note", help="an extra footer line, e.g. the dataset's licence and source")
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
    sug = sub.add_parser("suggest", parents=[run_dir], help="which training texts a person should label next")
    sug.add_argument("--n", type=int, default=100, help="how many texts")
    sug.add_argument("--out", default="to_label.jsonl", help="JSONL to write, with an empty label to fill in")
    sug.set_defaults(fn=cmd_suggest)
    ana = sub.add_parser(
        "analyse", parents=[run_dir], help="where the teacher and student go wrong (RUN/analysis.html)"
    )
    ana.add_argument("--top", type=int, default=15, help="how many confusions and disagreements to list")
    ana.set_defaults(fn=cmd_analyse)
    sub.add_parser(
        "status", parents=[run_dir], help="how far a run is, and whether it is still running"
    ).set_defaults(fn=cmd_status)
    cmp = sub.add_parser("compare", help="one markdown table comparing several runs (e.g. student sizes)")
    cmp.add_argument("runs", nargs="+", help="run directories or results.json files")
    cmp.set_defaults(fn=cmd_compare)
    sub.add_parser(
        "run", parents=[label, train, export, ev, common], help="label, train, export and eval"
    ).set_defaults(fn=cmd_run)
    return p


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    if getattr(a, "threads", None):
        import torch

        torch.set_num_threads(a.threads)
    a.fn(a)


if __name__ == "__main__":
    main()
