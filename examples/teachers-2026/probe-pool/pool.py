"""What the probe reads: the teacher's state at the last token (tinydistill's probe) against the mean of its
states over the text's tokens (as sentence embeddings from decoder LLMs often do, e.g. Springer et al. 2024,
arXiv 2402.15449), at the LM head's input and at a layer a quarter of the way back, which read better on two
of three tasks in examples/teachers-2026/probe-layers.sh.

For each run (labelled with --probe-only), the teacher is rebuilt as it labelled, with its prompt, and run
once over the gold and test texts, keeping every position's states. The probe (tinydistill's fit_probe, the
same gold texts as variants.py) is fit on each kind of state and scored on the test texts:

- last: the last token at the LM head's input, tinydistill's probe (a control: it should match the run);
- mean: the mean over the text's tokens (the text and the prompt after it, not the examples before it);
- last_mean: both, side by side;
- last_layer / mean_layer: the same at the output of layer ``--layer`` (default: -7 for 28 layers or fewer,
  else -12).

    python examples/teachers-2026/probe-pool/pool.py runs/trec50-gemma4-12b-probe-only-zeroshot \\
        > trec50.json

run once per run (a run's JSON list has one entry), for the five probe-only zero-shot runs into zeroshot.json
and the five Gemma 4 12B probe-only similar16 runs into similar16.json; table.py writes their tables. The
zero-shot runs used batches of 8192 tokens, the similar16 runs 4096 (TOKENS_PER_BATCH, for GPU memory).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tinydistill.cli import _run_members
from tinydistill.student import few_gold
from tinydistill.task import load_task
from tinydistill.teacher import PROBE_L2, fit_probe, probe_scores, text_hash

TOKENS_PER_BATCH = 4096


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    return float(
        np.mean(
            [2 * np.sum((p == c) & (y == c)) / max(np.sum(p == c) + np.sum(y == c), 1) for c in np.unique(y)]
        )
    )


@torch.no_grad()
def states(scorer, texts: list[str], layer: int) -> dict[str, torch.Tensor]:
    """Per kind of state, (texts, hidden): the last token and the mean over each text's own tokens, at the LM
    head's input and at the output of decoder layer ``layer``."""
    model, pad = scorer.model, scorer.tok.pad_token_id or 0
    n_layers = len(scorer._layers())
    rows = []
    for text, head in zip(texts, scorer.heads(texts), strict=True):
        ids = scorer.prefix_ids + scorer.text_ids(text, head)
        rows.append((ids, len(scorer.text_ids(text))))  # the text and the prompt after it
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][0]))
    out: dict[str, list[torch.Tensor]] = {k: [] for k in ("last", "mean", "last_layer", "mean_layer")}
    start = 0
    while start < len(order):
        width = len(rows[order[start]][0])
        end = start + 1
        while end < len(order) and (end - start + 1) * len(rows[order[end]][0]) <= TOKENS_PER_BATCH:
            end += 1
        batch = [rows[i] for i in order[start:end]]
        width = max(len(ids) for ids, _ in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        own = torch.zeros((len(batch), width), dtype=torch.bool)
        for r, (row, n_own) in enumerate(batch):  # left padding: every text ends at the last position
            ids[r, width - len(row) :] = torch.tensor(row)
            mask[r, width - len(row) :] = 1
            own[r, width - n_own :] = True
        mask, own = mask.to(scorer.device), own.to(scorer.device)
        result = model(
            input_ids=ids.to(scorer.device),
            attention_mask=mask,
            position_ids=(mask.cumsum(1) - 1).clamp_min(0),
            output_hidden_states=True,
            logits_to_keep=1,
        )
        # hidden_states: the embeddings, then each layer's output; the last one after the final norm, which
        # is the LM head's input
        final, mid = result.hidden_states[-1].float(), result.hidden_states[n_layers + layer + 1].float()
        weights = own.float()[..., None] / own.sum(1)[:, None, None]
        out["last"].append(final[:, -1].cpu())
        out["mean"].append((final * weights).sum(1).cpu())
        out["last_layer"].append(mid[:, -1].cpu())
        out["mean_layer"].append((mid * weights).sum(1).cpu())
        start = end
    back = torch.tensor(order).argsort()
    got = {k: torch.cat(v)[back] for k, v in out.items()}
    got["last_mean"] = torch.cat([got["last"], got["mean"]], 1)
    return got


def pool(run: Path, layer: int | None) -> dict:
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    task = load_task(run / "task.yaml")
    train = [json.loads(line) for line in (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    test = [json.loads(line) for line in (run / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    tests = {r["text"] for r in test}
    chosen = few_gold([r["gold"] for r in train], meta["gold_per_class"], 0)
    gold = list(
        {
            r["text"]: g
            for r, g in zip(train, chosen, strict=True)
            if g is not None and r["text"] not in tests
        }.items()
    )
    shots = run / "examples.jsonl"  # the gold examples the prompts drew from, as the run labelled
    rows = []
    if shots.exists():
        rows = [json.loads(line) for line in shots.read_text(encoding="utf-8").splitlines()]
    examples = [(r["text"], r["gold"]) for r in rows] or None
    member = _run_members(run, meta, task, meta["teacher"], "cuda", examples)[0]
    scorer = getattr(member, "base", member)
    n_layers = len(scorer._layers())
    layer = layer if layer is not None else (-7 if n_layers <= 28 else -12)
    t0 = time.perf_counter()
    texts = [t for t, _ in gold] + [r["text"] for r in test]
    x = states(scorer, texts, layer)
    seconds = time.perf_counter() - t0
    yg, yt = [g for _, g in gold], np.array([r["gold"] for r in test])
    # the control: the last-token states against those the run cached while labelling
    with np.load(f"runs/cache/teacher-{meta['fingerprint']}.states.npz") as z:
        cached = dict(zip(z["h"].tolist(), z["x"], strict=True))
    pairs = [(x["last"][i], cached[text_hash(t)]) for i, t in enumerate(texts) if text_hash(t) in cached]
    cos = torch.nn.functional.cosine_similarity
    cosine = [float(cos(a, torch.from_numpy(b).float(), 0)) for a, b in pairs]
    out: dict = {
        "run": run.name,
        "teacher": meta["teacher"],
        "layer": layer,
        "n_layers": n_layers,
        "n_gold": len(gold),
        "n_test": len(test),
        "seconds": round(seconds, 1),
        "cached_states_compared": len(pairs),
        "cached_states_min_cosine": min(cosine) if cosine else None,
    }
    for name, xs in x.items():
        w, b = fit_probe(xs[: len(gold)], yg, len(task.labels), False, PROBE_L2)
        predicted = np.array(probe_scores(w, b, xs[len(gold) :], False)).argmax(1)
        out[name] = {"accuracy": float(np.mean(predicted == yt)), "macro_f1": macro_f1(yt, predicted)}
    del scorer, member
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    layer = None
    if "--layer" in args:
        i = args.index("--layer")
        layer = int(args[i + 1])
        args = args[:i] + args[i + 2 :]
    print(json.dumps([pool(Path(a), layer) for a in args], indent=1))
