"""Command-line interface: label -> train -> export -> eval, or `run` for all of them."""

from __future__ import annotations

import argparse
import json
import math
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


def _softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    e = [math.exp(s - m) for s in scores]
    z = sum(e)
    return [x / z for x in e]


def _argmax(xs) -> int:
    return max(range(len(xs)), key=xs.__getitem__)


# ---------- steps ----------


def cmd_label(a: argparse.Namespace) -> None:
    from .data import read_examples
    from .teacher import ScoreCache, iter_unique, load_teacher

    task = load_task(a.task)
    run = Path(a.out)
    run.mkdir(parents=True, exist_ok=True)
    (run / "task.yaml").write_text(
        yaml.safe_dump(task.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    log(f"loading teacher {a.teacher}")
    examples = None
    if a.few_shot and "{examples}" not in task.prompt:
        raise SystemExit("--few-shot needs an {examples} line in the task's prompt")
    if a.few_shot:
        from .data import read_examples
        from .student import few_shot_examples

        rows = list(read_examples(a.data, task, a.limit))
        if not any(e.gold is not None for e in rows):
            raise SystemExit("--few-shot needs gold labels in the training data")
        examples = few_shot_examples([e.text for e in rows], [e.gold for e in rows], a.few_shot, a.seed)
        with (run / "examples.jsonl").open("w", encoding="utf-8") as f:
            for text, y in examples:
                f.write(json.dumps({"text": text, "gold": y}) + "\n")
        log(f"few-shot prompt with {len(examples)} gold examples ({a.few_shot} per class)")
    scorer = load_teacher(a.teacher, task, a.device, a.load_4bit, a.teacher_tokens, examples)
    fp = scorer.fingerprint()
    cache_dir = Path(a.cache_dir) if a.cache_dir else run
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = ScoreCache(cache_dir / f"teacher-{fp}.jsonl")
    if cache.scores:
        log(f"resuming: {len(cache.scores)} texts already scored")
    seconds, scored = 0.0, 0
    for split, spec, limit in (("train", a.data, a.limit), ("test", a.test, a.test_limit)):
        if not spec:
            continue
        t0 = time.perf_counter()

        def progress(n: int, split=split, t0=t0) -> None:
            if n % 100 == 0:
                log(f"{split}: scored {n} new texts ({n / (time.perf_counter() - t0):.1f}/s)")

        texts = iter_unique(e.text for e in read_examples(spec, task, limit, with_gold=False))
        n = cache.fill(scorer, texts, progress)
        seconds += time.perf_counter() - t0
        scored += n
        rows = 0
        with (run / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for e in read_examples(spec, task, limit):
                f.write(json.dumps({"text": e.text, "gold": e.gold, "scores": cache.get(e.text)}) + "\n")
                rows += 1
        log(f"{split}: {rows} texts ({n} scored now, the rest from the cache)")
    _meta(
        run,
        {
            "teacher": a.teacher,
            "few_shot": a.few_shot,
            "n_examples": len(examples or []),
            "load_4bit": a.load_4bit,
            "teacher_device": str(scorer.device),
            "fingerprint": fp,
            "data": a.data,
            "test": a.test,
            "label_seconds": round(seconds, 1),
            "label_texts_scored": scored,
            "teacher_tokens": a.teacher_tokens,
        },
    )


def cmd_train(a: argparse.Namespace) -> None:
    import torch

    from .student import few_gold, train_student
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
    gold = few_gold(all_gold, a.gold_per_class, a.seed) if a.gold_per_class else None
    n_gold = sum(g is not None for g in gold) if gold else 0
    device = pick_device(a.device)
    probs = torch.tensor([_softmax(r["scores"]) for r in rows])
    hp = (a.epochs, a.batch_size, a.lr, a.max_length, device, a.seed, log)

    log(f"distilling {a.student} on {len(rows)} texts, {n_gold} with gold labels ({device})")
    t0 = time.perf_counter()
    train_student(texts, probs, task.labels, run / "student", a.student, gold, a.ce_weight, *hp)
    meta = {
        "student": a.student,
        "epochs": a.epochs,
        "lr": a.lr,
        "batch_size": a.batch_size,
        "gold_per_class": a.gold_per_class,
        "n_gold": n_gold,
        "ce_weight": a.ce_weight if n_gold else 0.0,
        "student_device": device,
        "train_seconds": round(time.perf_counter() - t0, 1),
        "upper_bound": False,
    }
    if a.upper_bound:
        keep = [i for i, g in enumerate(all_gold) if g is not None]
        log(f"training the upper bound: {a.student} on all {len(keep)} gold training labels, no teacher")
        train_student(
            [texts[i] for i in keep],
            None,
            task.labels,
            run / "student-gold",
            a.student,
            [all_gold[i] for i in keep],
            0.0,
            *hp,
        )
        meta.update(upper_bound=True, n_gold_upper_bound=len(keep))
    _meta(run, meta)


def cmd_export(a: argparse.Namespace) -> None:
    from .export import export_onnx

    run = Path(a.run)
    for p in export_onnx(run / "student", run / "onnx", int8=a.int8):
        log(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")


def cmd_eval(a: argparse.Namespace) -> None:
    import torch

    from .evaluate import (
        accuracy,
        bench,
        dir_size_mb,
        environment,
        hf_weights_mb,
        macro_f1,
        markdown_table,
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
    teacher_pred = [_argmax(r["scores"]) for r in labelled]
    bench_texts = [r["text"] for r in (test or train)]
    if not bench_texts:
        raise SystemExit("no texts to evaluate on: label a --test set (or training data) first")
    threads = torch.get_num_threads()

    def quality(pred: list[int]) -> dict:
        if not gold:
            return {}
        return {
            "accuracy": accuracy(pred, gold),
            "macro_f1": macro_f1(pred, gold),
            "agreement": accuracy(pred, teacher_pred),
        }

    models: list[dict] = []
    teacher_q = quality(teacher_pred)
    teacher_q.pop("agreement", None)
    devices = [d for d in (a.teacher_devices or meta.get("teacher_device", "cpu")).split(",") if d]
    for dev in devices:
        row = {
            "role": "teacher",
            "name": "teacher, "
            + (f"{meta['few_shot']}-shot" if meta.get("few_shot") else "zero-shot")
            + (", 4-bit" if meta.get("load_4bit") else ""),
            "device": dev,
            **teacher_q,
            "size_mb": hf_weights_mb(meta["teacher"]),
        }
        if not a.skip_teacher_bench:
            log(f"benchmarking the teacher on {dev}")
            scorer = load_teacher(
                meta["teacher"],
                task,
                dev,
                meta.get("load_4bit", False),
                meta.get("teacher_tokens", 2048),
                [(r["text"], r["gold"]) for r in _read_jsonl(run / "examples.jsonl")] or None,
            )
            row["params_m"] = sum(p.numel() for p in scorer.model.parameters()) / 1e6
            row["size_mb"] = scorer.model.get_memory_footprint() / 1e6  # weights as loaded (e.g. 4-bit)
            score = scorer.score_many  # several texts are packed into shared forward passes
            row.update(bench(score, bench_texts, a.teacher_reps))
            if row["latency_ms_b1"] * 32 * 4 < 180_000:  # batch 32 only when it takes under three minutes
                row["latency_ms_b32"] = time_calls(score, bench_texts, 32, 3)
                row["throughput_per_s"] = 32 / (row["latency_ms_b32"] / 1000)
            del scorer
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()
        models.append(row)

    student_dir = run / "student"
    if student_dir.exists():
        log("evaluating the student (PyTorch, CPU)")
        model, tok = load_student(student_dir, "cpu")
        pred = student_logits(model, tok, texts).argmax(-1).tolist() if texts else []
        row = {
            "role": "student",
            "name": "student, PyTorch",
            "device": "cpu",
            **quality(pred),
            "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
            "size_mb": dir_size_mb(list(student_dir.glob("*.safetensors")) + list(student_dir.glob("*.bin"))),
        }
        row.update(
            bench(
                lambda b: student_logits(model, tok, b, batch_size=32),
                bench_texts,
                a.reps,
                max(3, a.reps // 5),
            )
        )
        models.append(row)
    gold_dir = run / "student-gold"
    if gold_dir.exists():
        log("evaluating the upper bound (PyTorch, CPU)")
        model, tok = load_student(gold_dir, "cpu")
        pred = student_logits(model, tok, texts).argmax(-1).tolist() if texts else []
        models.append(
            {
                "role": "upper bound",
                "name": "upper bound: student on all gold labels",
                "device": "cpu",
                **quality(pred),
            }
        )
    for int8 in (False, True):
        f = run / "onnx" / ("model.int8.onnx" if int8 else "model.onnx")
        if not f.exists():
            continue
        log(f"evaluating {f.name} (onnxruntime, CPU)")
        pr = Predictor(run / "onnx", int8=int8, threads=threads)
        pred = pr.logits(texts).argmax(-1).tolist() if texts else []
        row = {
            "role": "student",
            "name": f"student, ONNX {'int8' if int8 else 'fp32'}",
            "device": "cpu",
            **quality(pred),
            "size_mb": dir_size_mb([f]),
        }
        row.update(bench(lambda b, pr=pr: pr.logits(b), bench_texts, a.reps, max(3, a.reps // 5)))
        models.append(row)

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
                "n_gold",
                "upper_bound",
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
            )
        },
        "environment": environment(threads),
        "models": models,
    }
    (run / "results.json").write_text(json.dumps(results, indent=2))
    md = markdown_table(results)
    (run / "results.md").write_text(md)
    print(md)


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
    probs = pr.predict_proba(texts)
    for t, p in zip(texts, probs, strict=True):
        i = int(p.argmax())
        print(
            json.dumps(
                {"text": t, "label": pr.labels[i], "confidence": round(float(p[i]), 4)}, ensure_ascii=False
            )
        )


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
    label.add_argument("--teacher", default=DEFAULT_TEACHER, help="Hugging Face causal LM")
    label.add_argument(
        "--load-4bit", action="store_true", help="load the teacher in 4 bit (bitsandbytes, CUDA)"
    )
    label.add_argument(
        "--cache-dir",
        help="where teacher scores are cached (default: the run directory); "
        "runs that share it label each text once",
    )
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
        "--student", default="sentence-transformers/all-MiniLM-L6-v2", help="Hugging Face encoder"
    )
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--max-length", type=int, default=128)
    train.add_argument(
        "--gold-per-class",
        type=int,
        default=0,
        metavar="K",
        help="also use the gold labels of K training examples per class (cross-entropy term)",
    )
    train.add_argument("--ce-weight", type=float, default=1.0, help="weight of that cross-entropy term")
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
    pr.set_defaults(fn=cmd_predict)
    sub.add_parser(
        "run", parents=[label, train, export, ev, common], help="label, train, export and eval"
    ).set_defaults(fn=cmd_run)
    return p


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    if a.threads:
        import torch

        torch.set_num_threads(a.threads)
    a.fn(a)


if __name__ == "__main__":
    main()
