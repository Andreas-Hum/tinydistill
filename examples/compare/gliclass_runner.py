"""GLiClass zero-shot (knowledgator/gliclass-*-v3.0) on a task's test set, on the CPU: no gold labels and no
training. The labels are the task's label names, or with --descriptions "name: description" for every label
that has one in the task file. CLINC150 has no descriptions; there --descriptions describes only its "oos"
label, with the wording tinydistill uses for its `outside` label. Writes
examples/compare/results/gliclass-<task>-<size>[-descriptions].json.

Needs gliclass in its own environment (see examples/compare/run.sh).

    python examples/compare/gliclass_runner.py banking77 --size base --threads 16
"""

import argparse
import time

import common
import torch
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

from tinydistill.task import OUTSIDE

p = argparse.ArgumentParser()
p.add_argument("tasks", nargs="+", choices=list(common.TASKS))
p.add_argument("--size", default="base", choices=["base", "large"])
p.add_argument("--descriptions", action="store_true", help="add each label's description to its name")
p.add_argument("--threads", type=int, default=16, help="CPU threads for scoring the test set")
a = p.parse_args()

name_of = f"knowledgator/gliclass-{a.size}-v3.0"
model = GLiClassModel.from_pretrained(name_of)
tok = AutoTokenizer.from_pretrained(name_of, add_prefix_space=True)
pipe = ZeroShotClassificationPipeline(
    model, tok, classification_type="single-label", device="cpu", progress_bar=False
)
size_mb = sum(t.numel() * t.element_size() for t in model.state_dict().values()) / 1e6

for name in a.tasks:
    task, _, _, test_texts, test_gold = common.load(name)
    descs = list(task.descriptions or [None] * len(task.labels))
    if a.descriptions and name == "clinc150":
        descs[task.labels.index("oos")] = OUTSIDE[2]
    if a.descriptions and not any(descs):
        print(f"{name}: no label descriptions, skipped")
        continue
    names = [
        f"{v}: {d}" if a.descriptions and d else v for v, d in zip(task.verbalisations, descs, strict=True)
    ]
    index = {n: i for i, n in enumerate(names)}
    assert len(index) == len(names), "label names must be distinct"
    longest = max(len(pipe.pipe.prepare_inputs([t], names, True)["input_ids"][0]) for t in test_texts[:200])
    assert longest < pipe.pipe.max_length, f"{longest} tokens: the labels would be cut off"

    def predict(texts: list[str], batch_size: int = 1, names=names, index=index) -> list[int]:
        return [index[r[0]["label"]] for r in pipe(texts, names, batch_size=batch_size)]

    torch.set_num_threads(a.threads)
    t0 = time.perf_counter()
    pred = predict(test_texts, batch_size=8)
    seconds = time.perf_counter() - t0
    torch.set_num_threads(common.LATENCY_THREADS)
    latency = common.latency_ms(predict, test_texts)
    common.write(
        f"gliclass-{name}-{a.size}" + ("-descriptions" if a.descriptions else ""),
        {
            "task": name,
            "method": "GLiClass",
            "model": name_of,
            "short": a.size + (", names and descriptions" if a.descriptions else ""),
            "labels": names,
            "n_gold": 0,
            "n_test": len(test_texts),
            **common.scores(pred, test_gold),
            "test_seconds": seconds,
            "tokens_longest_input": longest,
            "latency_ms_b1": latency,
            "size_mb": size_mb,
            "device": "cpu",
            "test_threads": a.threads,
            "latency_threads": common.LATENCY_THREADS,
        },
    )
