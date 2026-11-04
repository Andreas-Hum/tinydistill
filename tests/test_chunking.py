import json

import numpy as np
import pytest
import torch
from conftest import TINY_ENCODER, TINY_LM
from test_cli import TASK, write_jsonl
from transformers import AutoTokenizer

from tinydistill.cli import main
from tinydistill.export import Predictor, export_onnx
from tinydistill.student import by_windows, load_student, student_logits, train_student, windows
from tinydistill.teacher import combine_scores

LONG = " ".join(["this was a good film and the plot was bad"] * 3)


def test_windows_cover_the_text_with_the_given_stride():
    tok = AutoTokenizer.from_pretrained(TINY_ENCODER)
    ids = tok(LONG, add_special_tokens=False)["input_ids"]
    n = len(ids)
    assert n > 40
    half = windows(tok, LONG, 8)  # default stride: half a window
    assert half == [tok.decode(ids[s : s + 8]) for s in range(0, n - 4, 4)]
    assert len(half) == -(-(n - 8) // 4) + 1  # the last window reaches the end, none starts past it
    apart = windows(tok, LONG, 8, 8)  # no overlap
    assert apart == [tok.decode(ids[s : s + 8]) for s in range(0, n, 8)]
    short = "this was"
    assert windows(tok, short, 8) == [short]  # unchanged, not decoded
    assert windows(tok, LONG, n) == [LONG]


def test_by_windows_averages_probabilities():
    rows = [[0.0, 1.0], [2.0, -1.0], [0.5, 0.5]]
    single = by_windows(lambda w: rows[: len(w)], ["a", "b"], lambda t: ["x", "y"] if t == "a" else ["z"])
    p = torch.tensor(rows[:2]).softmax(-1).mean(0)
    np.testing.assert_allclose(torch.tensor(single[0]).softmax(-1), p, atol=1e-6)
    assert single[1] == rows[2]  # a text that is one window keeps its scores
    multi = by_windows(lambda w: rows[:2], ["a"], lambda t: ["x", "y"], multi_label=True)
    np.testing.assert_allclose(torch.tensor(multi[0]).sigmoid(), torch.tensor(rows[:2]).sigmoid().mean(0))


def test_a_model_trained_on_windows_averages_them(tmp_path):
    texts = [LONG, "this was", "bad", "good film"] * 2
    probs = torch.tensor([[0.7, 0.3], [0.3, 0.7]] * 4)
    train_student(
        texts, probs, ["a", "b"], tmp_path / "s", TINY_ENCODER, epochs=1, max_length=4, chunk_tokens=8
    )
    model, tok = load_student(tmp_path / "s")
    c = model.config
    assert (c.tinydistill_chunk_tokens, c.tinydistill_chunk_stride, c.tinydistill_max_length) == (8, 4, 10)
    export_onnx(tmp_path / "s", tmp_path / "onnx")
    p = Predictor(tmp_path / "onnx")
    assert (p.chunk_tokens, p.chunk_stride, p.max_length) == (8, 4, 10)
    plain = Predictor(tmp_path / "onnx")
    plain.chunk_tokens = None  # reads each text as one input
    docs = [LONG, "this was", LONG[:30]]
    want = np.stack([plain.predict_proba(p.cut(d)).mean(0) for d in docs])
    np.testing.assert_allclose(p.predict_proba(docs), want, atol=1e-5)
    assert len(p.predict(docs)) == len(p.describe(docs)) == 3
    assert np.array_equal(p.logits(["this was"]), plain.logits(["this was"]))
    np.testing.assert_allclose(p.logits(docs), student_logits(model, tok, docs).numpy(), atol=1e-4)


def test_without_chunking_nothing_changes(tmp_path):
    texts = [LONG, "short"] * 4
    probs = torch.tensor([[0.7, 0.3], [0.3, 0.7]] * 4)
    train_student(texts, probs, ["a", "b"], tmp_path / "s", TINY_ENCODER, epochs=1, max_length=20)
    config = json.loads((tmp_path / "s" / "config.json").read_text(encoding="utf-8"))
    assert "tinydistill_chunk_tokens" not in config and config["tinydistill_max_length"] == 20
    export_onnx(tmp_path / "s", tmp_path / "onnx")
    p = Predictor(tmp_path / "onnx")
    assert p.chunk_tokens is None and p.cut(LONG) == [LONG]
    assert np.array_equal(p.logits(texts[:2]), p._logits(texts[:2], 32))


def test_label_train_and_eval_with_windows(tmp_path, capsys):
    (tmp_path / "task.yaml").write_text(TASK)
    words = ["good", "bad", "great", "awful"]
    docs = [f"{LONG} {w} {i}" for i, w in enumerate(words * 2)]
    gold = ["pos", "neg"] * 4
    rows = [{"review": d, "sentiment": g} for d, g in zip(docs, gold, strict=True)]
    write_jsonl(tmp_path / "train.jsonl", rows)
    write_jsonl(tmp_path / "test.jsonl", rows)
    run = tmp_path / "run"
    main(
        [
            *("run", str(tmp_path / "task.yaml"), "--out", str(run)),
            *("--data", str(tmp_path / "train.jsonl"), "--test", str(tmp_path / "test.jsonl")),
            *("--teacher", TINY_LM, "--student", TINY_ENCODER, "--chunk-tokens", "16"),
            *("--epochs", "1", "--reps", "2", "--teacher-reps", "2", "--device", "cpu", "--threads", "2"),
        ]
    )
    tok = AutoTokenizer.from_pretrained(TINY_ENCODER)
    train = [json.loads(line) for line in (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    test = [json.loads(line) for line in (run / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["text"] for r in train if r["doc"] == 0] == windows(tok, docs[0], 16, 8)
    assert {r["doc"] for r in train} == set(range(len(docs))) and len(train) > len(docs)
    assert [r["text"] for r in test] == docs  # one row per document
    for d, row in enumerate(test):  # the document's teacher scores: the mean of its windows' probabilities
        want = combine_scores([r["scores"] for r in train if r["doc"] == d])
        np.testing.assert_allclose(row["scores"], want, atol=1e-9)
        assert all(r["gold"] == d % 2 for r in train if r["doc"] == d)  # gold applies to every window
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    assert (meta["chunk_tokens"], meta["chunk_stride"], meta["chunk_tokenizer"]) == (16, 8, TINY_ENCODER)

    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert results["data"]["n_test"] == len(docs) and results["training"]["chunk_tokens"] == 16
    onnx = next(m for m in results["models"] if m["name"] == "student, ONNX fp32")
    pred = Predictor(run / "onnx", int8=False).predict(docs)
    assert onnx["accuracy"] == sum(p == g for p, g in zip(pred, gold, strict=True)) / len(docs)
    assert "windows of that many tokens, 8 apart" in (run / "results.md").read_text(encoding="utf-8")
    capsys.readouterr()
    main(["web", str(run / "onnx"), "--out", str(tmp_path / "site")])
    assert "the page reads only the first one" in capsys.readouterr().err


def test_few_shot_similar_is_refused_with_windows(tmp_path):
    # a window is not its own document's text, so the document and its gold label would be its examples
    (tmp_path / "task.yaml").write_text(TASK.replace("  {labels}\n", "  {labels}\n  {examples}\n"))
    write_jsonl(tmp_path / "d.jsonl", [{"review": f"{LONG} {i}", "sentiment": "pos"} for i in range(2)])
    args = [
        "label",
        str(tmp_path / "task.yaml"),
        "--data",
        str(tmp_path / "d.jsonl"),
        "--out",
        str(tmp_path / "r"),
    ]
    with pytest.raises(SystemExit, match="--chunk-tokens"):
        main(
            [
                *args,
                "--teacher",
                TINY_LM,
                "--chunk-tokens",
                "16",
                "--few-shot-similar",
                "1",
                "--device",
                "cpu",
            ]
        )
