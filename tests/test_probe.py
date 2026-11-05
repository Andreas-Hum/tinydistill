import argparse
import json
import math
import random

import numpy as np
import pytest
import torch
from conftest import TINY_ENCODER, TINY_LM
from test_cli import TASK, write_jsonl
from test_scoring import POOL, SHOTS, hybrid, lm, sliding, unpackable  # noqa: F401  (fixtures)

from tinydistill.cli import _probe, _run_teacher, main
from tinydistill.teacher import (
    Ensemble,
    LabelScorer,
    ProbeScorer,
    ScoreCache,
    combine_scores,
    fit_probe,
    probe_scores,
    text_hash,
)

TEXTS = ["great product, would buy again", "", "meh", "bad", "a much longer review of the thing", "ok then"]


def close(got, want, atol):
    np.testing.assert_allclose(np.array(got), np.array(want), atol=atol, rtol=0)


@torch.no_grad()
def plain_state(scorer: LabelScorer, text: str) -> list[float]:
    """The last layer's normed hidden state at the text's last token, from one plain forward pass over
    prompt + text (no cache, no packing)."""
    ids = torch.tensor([scorer.prefix_ids + scorer.text_ids(text)])
    return scorer.model.base_model(input_ids=ids).last_hidden_state[0, -1].tolist()


def check_states(scorer: LabelScorer, texts: list[str]) -> None:
    plain = scorer.form_scores_many(texts)
    scorer.keep_states = True
    kept = scorer.form_scores_many(texts)
    assert torch.equal(plain, kept)  # the scores are unchanged
    assert set(scorer.states) == set(texts)
    for t in texts:
        assert scorer.states[t].tolist() == pytest.approx(plain_state(scorer, t), abs=1e-4)


@pytest.mark.parametrize("max_tokens", [12, 10_000])  # several packs, or one
def test_packed_states_equal_a_plain_forward_pass(lm, task, max_tokens):  # noqa: F811
    model, tok = lm
    check_states(LabelScorer(model, tok, task, max_tokens=max_tokens), TEXTS)
    # no suffix: an empty text's state is the prompt's last one
    bare = type(task)(**{**task.__dict__, "prompt": "Is this review positive?\n{labels}\n{text}"})
    check_states(LabelScorer(model, tok, bare, use_chat_template=False), ["", "fine", ""])


@pytest.mark.parametrize("teacher", [unpackable("hybrid"), unpackable("sliding")])
def test_batched_states_equal_a_plain_forward_pass(teacher, task, request):
    model, tok = request.getfixturevalue(teacher)
    shots = type(task)(**{**task.__dict__, "prompt": SHOTS})
    scorer = LabelScorer(model, tok, shots, examples=POOL, max_tokens=12)
    scorer._encode_prefix()
    assert not scorer._packable
    check_states(scorer, TEXTS)


def test_states_do_not_change_the_fingerprint(lm, task):  # noqa: F811
    scorer = LabelScorer(*lm, task)
    before = scorer.fingerprint()
    scorer.keep_states = True
    assert scorer.fingerprint() == before


def test_the_probe_fits_separable_data():
    g = torch.Generator().manual_seed(0)
    centres = torch.randn(3, 16, generator=g) * 5
    y = [i % 3 for i in range(60)]
    x = centres[y] + torch.randn(60, 16, generator=g) + 100  # far from the origin: standardised for the fit
    w, b = fit_probe(x, y, 3)
    assert w.shape == (3, 16) and b.shape == (3,)
    scores = probe_scores(w, b, x, False)
    assert [max(range(3), key=s.__getitem__) for s in scores] == y
    assert all(math.isclose(sum(map(math.exp, s)), 1, abs_tol=1e-6) for s in scores)
    assert torch.equal(fit_probe(x, y, 3)[0], w)  # deterministic
    # multi-label: one sigmoid per label; label 2 is on when label 0 or 1 is
    sets = [[c, 2] if c < 2 else [] for c in y]
    w, b = fit_probe(x, sets, 3, multi_label=True)
    got = [sorted(i for i, z in enumerate(s) if z > 0) for s in probe_scores(w, b, x, True)]
    assert got == [sorted(s) for s in sets]


def test_a_probe_in_an_ensemble_reuses_its_base_pass(lm, task):  # noqa: F811
    model, tok = lm
    base = LabelScorer(model, tok, task)
    torch.manual_seed(0)
    probe = ProbeScorer(base, torch.randn(3, model.config.hidden_size), torch.randn(3))
    base.score_many(["warm up"])  # the prompt's cache
    calls = []
    hook = model.register_forward_hook(lambda *_: calls.append(1))
    try:
        alone = base.score_many(TEXTS)
        n = len(calls)
        both = Ensemble([base, probe]).score_many(TEXTS)
    finally:
        hook.remove()
    assert len(calls) == 2 * n  # base's pass only
    mine = probe.score_many(TEXTS)
    want = probe_scores(probe.weight, probe.bias, torch.stack([base.states[t] for t in TEXTS]), False)
    close(mine, want, 1e-9)
    for got, a, p in zip(both, alone, mine, strict=True):
        assert got == pytest.approx(combine_scores([a, p]), abs=1e-9)
    close(ProbeScorer(LabelScorer(model, tok, task), probe.weight, probe.bias).score_many(TEXTS), mine, 1e-6)
    # alone, it runs its base itself


class StateScorer:
    def __init__(self):
        self.calls, self.states = [], {}

    def score_many(self, texts):
        self.calls.extend(texts)
        self.states = {t: torch.full((4,), float(len(t))) for t in texts}
        return [[float(len(t)), -1.0] for t in texts]


def test_states_are_cached_next_to_the_scores(tmp_path):
    path = tmp_path / "teacher-x.jsonl"
    ScoreCache(path).fill(StateScorer(), ["a", "bb"])  # scores only, as without --probe
    scorer = StateScorer()
    cache = ScoreCache(path, states=True)
    assert cache.fill(scorer, ["a", "bb", "ccc"]) == 3  # the cached texts again, for their states
    assert len(path.read_text().splitlines()) == 3  # but their score rows are not written twice
    assert cache.states_path == tmp_path / "teacher-x.states.npz"
    again = StateScorer()
    cache = ScoreCache(path, states=True)
    assert cache.fill(again, ["a", "bb", "ccc"]) == 0 and again.calls == []
    assert cache.state("ccc").dtype == np.float16 and cache.state("ccc").tolist() == [3.0] * 4
    with np.load(cache.states_path) as z:
        assert sorted(z["h"].tolist()) == sorted(text_hash(t) for t in ["a", "bb", "ccc"])
    assert ScoreCache(path).fill(StateScorer(), ["a", "bb", "ccc"]) == 0  # without states: all cached


class FakeCache:
    def __init__(self, scores, states):
        self.scores, self.states = scores, states

    def get(self, text):
        return self.scores[text]

    def state(self, text):
        return self.states[text]


def test_the_dev_report_holds_out_gold_texts(tmp_path):
    rng = np.random.default_rng(0)
    gold = [(f"t{i}", i % 2) for i in range(20)]
    states = {t: (rng.normal(size=8) + 4 * y).astype(np.float16) for t, y in gold}
    always_0 = {t: [0.0, -2.0] for t, _ in gold}  # the in-context teacher says 0 every time
    a = argparse.Namespace(
        probe_only=False, probe_l2=1e-3, probe_dev=0.25, gold_per_class=10, gold_file=None, seed=0
    )
    task = type("T", (), {"multi_label": False, "labels": ["a", "b"]})()
    scores, entry = _probe(a, task, [FakeCache(always_0, states)], gold, [t for t, _ in gold], tmp_path)
    dev = entry["dev"]
    assert (dev["n_fit"], dev["n_dev"]) == (15, 5) and entry["n_gold"] == 20
    order = list(range(20))
    random.Random(0).shuffle(order)  # as _probe holds them out (--seed 0)
    held = [gold[i] for i in order[:5]]
    assert dev["in_context"] == sum(y == 0 for _, y in held) / 5  # right where the gold label is 0
    assert dev["probe"] == 1.0 and dev["combined"] >= dev["in_context"]
    fit = torch.load(tmp_path / "probe.pt", weights_only=True)
    x = torch.from_numpy(np.stack([states[t] for t, _ in gold]).astype(np.float32))
    close([scores[text_hash(t)] for t, _ in gold], probe_scores(fit["weight"], fit["bias"], x, False), 1e-9)
    a.probe_dev = 0
    assert _probe(a, task, [FakeCache(always_0, states)], gold, [], tmp_path)[1]["dev"] is None


@pytest.fixture
def data(tmp_path):
    (tmp_path / "task.yaml").write_text(TASK)
    words = ["good", "bad", "great", "awful", "fine", "terrible", "nice", "poor"]
    rows = [
        {"review": f"it was {w} {i}", "sentiment": "pos" if i % 2 == 0 else "neg"}
        for i, w in enumerate(words * 3)
    ]
    write_jsonl(tmp_path / "train.jsonl", rows)
    write_jsonl(tmp_path / "test.jsonl", [{"review": f"so {w}", "sentiment": "pos"} for w in words[:4]])
    return [str(tmp_path / "task.yaml"), "--data", str(tmp_path / "train.jsonl")] + [
        "--test",
        str(tmp_path / "test.jsonl"),
        "--teacher",
        TINY_LM,
        "--device",
        "cpu",
        "--cache-dir",
        str(tmp_path / "cache"),
    ]


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_label_with_a_probe_and_the_eval_round_trip(tmp_path, data, capsys):
    plain, run = tmp_path / "plain", tmp_path / "run"
    main(["label", *data, "--out", str(plain)])
    main(
        ["run", *data, "--out", str(run), "--probe", "--gold-per-class", "5", "--student", TINY_ENCODER]
        + ["--epochs", "1", "--batch-size", "4", "--reps", "2", "--teacher-reps", "2"]
    )
    err = capsys.readouterr().err
    assert "probe: teacher accuracy on 2 held-out gold texts (fit on 8)" in err
    assert "every text is in the cache" not in err  # the cached scores have no states yet
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    assert (
        meta["fingerprints"] == json.loads((plain / "meta.json").read_text(encoding="utf-8"))["fingerprints"]
    )
    probe = meta["probe"]
    assert probe["n_gold"] == 10 and probe["gold_per_class"] == 5 and not probe["only"]
    assert (probe["dev"]["n_fit"], probe["dev"]["n_dev"]) == (8, 2)
    assert all(0 <= probe["dev"][k] <= 1 for k in ("in_context", "probe", "combined"))
    fp = meta["fingerprint"]
    assert len(read(tmp_path / "cache" / f"teacher-{fp}.jsonl")) == 28  # each text's score row once
    cache = ScoreCache(tmp_path / "cache" / f"teacher-{fp}.jsonl", states=True)
    fit = torch.load(run / "probe.pt", weights_only=True)
    for split in ("train", "test"):
        rows, before = read(run / f"{split}.jsonl"), read(plain / f"{split}.jsonl")
        for r, b in zip(rows, before, strict=True):
            in_context, mine = r["teacher_scores"]
            assert in_context == b["scores"]  # the in-context scores are unchanged
            x = torch.from_numpy(cache.state(r["text"]).astype(np.float32))[None]
            assert mine == pytest.approx(probe_scores(fit["weight"], fit["bias"], x, False)[0], abs=1e-4)
            assert r["scores"] == pytest.approx(combine_scores(r["teacher_scores"]), abs=1e-12)
    # eval and the cascade rebuild the same teacher from meta.json and probe.pt
    test = read(run / "test.jsonl")
    got = _run_teacher(run, "cpu").score_many([r["text"] for r in test])
    for r, g in zip(test, got, strict=True):
        assert g == pytest.approx(r["scores"], abs=1e-4)
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert results["training"]["probe"] == probe
    assert results["models"][0]["name"] == "teacher, zero-shot, probe"
    md = (run / "results.md").read_text(encoding="utf-8")
    assert "averaged with those of a linear probe on its hidden states, fit on 10 gold training texts" in md
    # everything cached, states too: the teacher is not loaded again
    main(["label", *data, "--out", str(tmp_path / "again"), "--probe", "--gold-per-class", "5"])
    assert "every text is in the cache, so the teacher is not loaded" in capsys.readouterr().err
    assert read(tmp_path / "again" / "train.jsonl") == read(run / "train.jsonl")


def test_probe_only(tmp_path, data):
    run = tmp_path / "run"
    main(["label", *data, "--out", str(run), "--probe-only", "--gold-per-class", "3", "--probe-dev", "0"])
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    assert meta["probe"]["only"] and meta["probe"]["dev"] is None and meta["probe"]["n_gold"] == 6
    test = read(run / "test.jsonl")
    assert all("teacher_scores" not in r for r in test)
    teacher = _run_teacher(run, "cpu")
    assert isinstance(teacher, ProbeScorer)
    for r, g in zip(test, teacher.score_many([r["text"] for r in test]), strict=True):
        assert g == pytest.approx(r["scores"], abs=1e-4)
    main(["label", *data, "--out", str(run)])  # labelled again without --probe
    assert not (run / "probe.pt").exists()
    assert json.loads((run / "meta.json").read_text(encoding="utf-8"))["probe"] is None


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--gold-per-class", "2", "--api-url", "http://localhost:1/v1"], "needs a local teacher"),
        (["--gold-per-class", "2", "--vllm-url", "http://localhost:1/v1"], "needs a local teacher"),
        (["--gold-per-class", "2", "--chunk-tokens", "8"], "--chunk-tokens"),
        ([], "needs gold labels to fit on"),
        (["--gold-per-class", "2", "--probe-dev", "1"], "--probe-dev"),
    ],
)
def test_probe_errors(tmp_path, data, extra, message):
    with pytest.raises(SystemExit, match=message):
        main(["label", *data, "--out", str(tmp_path / "r"), "--probe", *extra])


def test_probe_only_needs_one_teacher_and_prompt(tmp_path, data):
    with pytest.raises(SystemExit, match="one teacher"):
        main(
            ["label", *data, "--out", str(tmp_path / "r"), "--probe-only", "--gold-per-class", "2"]
            + ["--prompts", data[0]]
        )
