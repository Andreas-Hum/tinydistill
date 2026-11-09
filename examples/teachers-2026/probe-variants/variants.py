"""Two variants of the probe, tried on the cached teacher states of probe-only runs (no GPU), against the
probe as it is (fit on the gold labels, L2 toward zero):

- self-training: fit on the gold, label the run's other training texts, refit on the gold plus the most
  confident 10, 30, 50 or 100% of them (Colombo et al., arXiv 2310.13998; arXiv 2503.03062);
- the label names as a prior: start from the teacher's LM-head rows for each label's tokens (its first token,
  or the mean over its tokens), times 0.5, 1 or 2, and pull the weights toward them instead of toward zero
  (as CLAP, arXiv 2312.12730, does for CLIP). The LM head is tied to the input embeddings in Gemma 4, whose
  rows are read from the checkpoint; the first two lines give these rows alone as a zero-shot classifier.

The probe teacher's test accuracy and macro-F1 for each, as JSON:

    python examples/teachers-2026/probe-variants/variants.py runs/banking77-gemma4-12b-probe-only-similar16 \\
        ... > examples/teachers-2026/probe-variants/variants.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

from tinydistill.student import few_gold
from tinydistill.task import load_task
from tinydistill.teacher import PROBE_L2, PROBE_STEPS, fit_probe, probe_scores, text_hash


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    return float(
        np.mean(
            [2 * np.sum((p == c) & (y == c)) / max(np.sum(p == c) + np.sum(y == c), 1) for c in np.unique(y)]
        )
    )


def fit_toward(x: torch.Tensor, gold: list[int], rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fit_probe with the L2 penalty toward ``rows`` (labels, hidden; in raw units) instead of toward zero."""
    x = x.double()
    mu, sd = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    z, target = (x - mu) / sd, (rows.double() * sd).detach()
    w, b = (
        target.clone().requires_grad_(True),
        torch.zeros(len(rows), dtype=torch.float64, requires_grad=True),
    )
    y = torch.tensor(gold)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=PROBE_STEPS, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z @ w.T + b, y) + PROBE_L2 * (w - target).pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    weight = (w / sd).detach()
    return weight.float(), (b - weight @ mu).detach().float()


def label_rows(model: str, verbalisations: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Each label's first-token row and mean token row of the (tied) LM head."""
    snap = snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.jinja"])
    tok = AutoTokenizer.from_pretrained(snap)
    ids = [tok(v, add_special_tokens=False)["input_ids"] for v in verbalisations]
    with safe_open(f"{snap}/model.safetensors", "pt") as f:
        table = f.get_slice("model.language_model.embed_tokens.weight")
        first = torch.stack([table[i[0] : i[0] + 1][0].float() for i in ids])
        mean = torch.stack([torch.stack([table[j : j + 1][0].float() for j in i]).mean(0) for i in ids])
    return first, mean


def variants(run: Path) -> dict:
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    task = load_task(run / "task.yaml")
    with np.load(f"runs/cache/teacher-{meta['fingerprint']}.states.npz") as z:
        states = dict(zip(z["h"].tolist(), z["x"], strict=True))
    train = [json.loads(line) for line in (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    test = [json.loads(line) for line in (run / "test.jsonl").read_text(encoding="utf-8").splitlines()]

    def st(texts: list[str]) -> torch.Tensor:
        return torch.from_numpy(np.stack([states[text_hash(t)] for t in texts]).astype(np.float32))

    tests = {r["text"] for r in test}
    chosen = few_gold([r["gold"] for r in train], meta["gold_per_class"], 0)
    gold = list(
        {
            r["text"]: g
            for r, g in zip(train, chosen, strict=True)
            if g is not None and r["text"] not in tests
        }.items()
    )
    seen = {t for t, _ in gold} | tests
    pool = list(dict.fromkeys(r["text"] for r in train if r["text"] not in seen))
    xg, yg = st([t for t, _ in gold]), [g for _, g in gold]
    xt, yt = st([r["text"] for r in test]), np.array([r["gold"] for r in test])
    out: dict = {
        "run": run.name,
        "teacher": meta["teacher"],
        "n_gold": len(gold),
        "n_pool": len(pool),
        "n_test": len(test),
    }

    def score(name: str, predicted: np.ndarray) -> None:
        out[name] = {"accuracy": float(np.mean(predicted == yt)), "macro_f1": macro_f1(yt, predicted)}

    w, b = fit_probe(xg, yg, len(task.labels), False, PROBE_L2)
    score("probe", np.array(probe_scores(w, b, xt, False)).argmax(1))
    confidence, label = torch.tensor(probe_scores(w, b, st(pool), False)).max(1)
    order, xp = confidence.argsort(descending=True), st(pool)
    for share in (0.1, 0.3, 0.5, 1.0):
        keep = order[: int(share * len(pool))]
        w2, b2 = fit_probe(
            torch.cat([xg, xp[keep]]), yg + label[keep].tolist(), len(task.labels), False, PROBE_L2
        )
        score(f"self_training_{round(share * 100)}", np.array(probe_scores(w2, b2, xt, False)).argmax(1))
    first, mean = label_rows(meta["teacher"], task.verbalisations)
    for name, rows in (("first", first), ("mean", mean)):
        score(f"label_rows_{name}_zero_shot", (xt @ rows.T).argmax(1).numpy())
        for scale in (0.5, 1.0, 2.0):
            w2, b2 = fit_toward(xg, yg, scale * rows)
            score(f"name_prior_{name}_x{scale}", np.array(probe_scores(w2, b2, xt, False)).argmax(1))
    return out


if __name__ == "__main__":
    print(json.dumps([variants(Path(a)) for a in sys.argv[1:]], indent=1))
