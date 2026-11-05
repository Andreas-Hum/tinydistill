"""What the comparison runners share: the four tasks, the 8-per-class gold that every method gets, the test
set, latency at batch 1 with 2 threads, and the results files in examples/compare/results/."""

import json
import statistics
from pathlib import Path

from tinydistill.data import read_examples
from tinydistill.evaluate import accuracy, macro_f1, time_calls
from tinydistill.student import few_gold
from tinydistill.task import load_task

HERE = Path(__file__).parent
EXAMPLES = HERE.parent
RESULTS = HERE / "results"
GOLD_PER_CLASS, GOLD_SEED, SEEDS = 8, 0, (0, 1, 2)  # as tinydistill's --gold-per-class 8, --seed 0, --seeds 3
LATENCY_TEXTS, LATENCY_THREADS = 100, 2

# task: (train, test), as the task's run.sh passes them to --data and --test
TASKS = {
    "banking77": ("hf:PolyAI/banking77:train", "hf:PolyAI/banking77:test"),
    "clinc150": ("hf:clinc/clinc_oos:plus:train", "hf:clinc/clinc_oos:plus:test"),
    "massive-da": ("hf:mteb/amazon_massive_intent:da:train", "hf:mteb/amazon_massive_intent:da:test"),
    "financial-tweets": (
        "hf:zeroshot/twitter-financial-news-sentiment:train",
        "hf:zeroshot/twitter-financial-news-sentiment:validation",
    ),
}


def load(name: str):
    """The task, its gold texts and labels (the same ones as `tinydistill run --gold-per-class 8 --seed 0`),
    and the test texts and labels."""
    task = load_task(EXAMPLES / name / "task.yaml")
    train_spec, test_spec = TASKS[name]
    train = list(read_examples(train_spec, task))
    test = list(read_examples(test_spec, task))
    gold = few_gold([e.gold for e in train], GOLD_PER_CLASS, GOLD_SEED)
    picked = [(e.text, g) for e, g in zip(train, gold, strict=True) if g is not None]
    return (
        task,
        [t for t, _ in picked],
        [g for _, g in picked],
        [e.text for e in test],
        [e.gold for e in test],
    )


def scores(pred: list[int], gold: list[int]) -> dict:
    return {"accuracy": accuracy(pred, gold), "macro_f1": macro_f1(pred, gold)}


def latency_ms(predict_one, texts: list[str]) -> float:
    """Median ms of one text at a time, over three passes through the first 100 test texts (after a warm-up).
    The caller sets the library to 2 threads first."""
    return time_calls(predict_one, texts[:LATENCY_TEXTS], 1, 3 * LATENCY_TEXTS)


def over_seeds(runs: list[dict]) -> dict:
    """Mean, standard deviation and per-seed values of every number in the per-seed runs."""
    out = {}
    for key in runs[0]:
        values = [r[key] for r in runs]
        out[key] = statistics.fmean(values)
        out[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        out[f"{key}_seeds"] = values
    return out


def write(name: str, result: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {path}: accuracy {result['accuracy']:.3f}, {result.get('latency_ms_b1', 0):.1f} ms per text"
    )
