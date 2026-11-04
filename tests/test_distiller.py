import numpy as np
import pytest
from conftest import TINY_ENCODER, TINY_LM

import tinydistill.teacher
from tinydistill import Distiller, Predictor
from tinydistill.distiller import NotFittedError
from tinydistill.task import Task

TEXTS = [f"this was {w} {i}" for i, w in enumerate(["good", "bad", "great", "awful", "fine", "terrible"] * 2)]
SMALL = {"teacher": TINY_LM, "student": TINY_ENCODER, "epochs": 1, "batch_size": 4, "device": "cpu"}


def quiet(msg: str) -> None:
    pass


def test_fit_predict_save_load_and_cache(tmp_path, task, monkeypatch):
    d = Distiller(task, cache_dir=tmp_path / "cache", log=quiet, **SMALL)
    gold = ["pos", None, 1, "neutral"] + [None] * (len(TEXTS) - 4)  # ids or indices, None for no label
    assert d.fit(TEXTS, gold) is d
    assert d.classes_ == task.labels
    assert d.teacher_scores_.shape == (len(TEXTS), 3)

    pred = d.predict(TEXTS)
    assert len(pred) == len(TEXTS) and all(p in task.labels for p in pred)
    proba = d.predict_proba(TEXTS)
    assert isinstance(proba, np.ndarray) and proba.shape == (len(TEXTS), 3)
    assert np.allclose(proba.sum(-1), 1, atol=1e-5)
    described = d.describe(TEXTS[:2])
    assert [x["label"] for x in described] == pred[:2]
    acc = d.score(TEXTS, pred)
    assert acc == 1.0 and isinstance(acc, float)
    assert d.score(TEXTS[:1], [task.labels.index(pred[0])]) == 1.0  # gold as an index

    out = d.save(tmp_path / "model")
    assert (out / "model.int8.onnx").exists() and (out / "task.yaml").exists()
    back = Distiller.load(out)
    assert back.classes_ == task.labels and back.predict(TEXTS) == pred
    assert np.allclose(back.predict_proba(TEXTS), proba)
    assert Predictor(out).predict(TEXTS) == pred
    run_onnx = tmp_path / "onnx"  # an exported folder without task.yaml, as `tinydistill export` writes
    d.save(run_onnx)
    (run_onnx / "task.yaml").unlink()
    assert Distiller.load(run_onnx).predict(TEXTS) == pred

    # The cache, in the files `tinydistill label --cache-dir` uses: a second Distiller needs no teacher.
    (cache,) = (tmp_path / "cache").glob("teacher-*.jsonl")
    assert len(cache.read_text(encoding="utf-8").splitlines()) == len(TEXTS)
    assert (tmp_path / "cache" / "teacher-keys.json").exists()

    def no_teacher(*args, **kwargs):
        raise AssertionError("the teacher was loaded although every text is cached")

    monkeypatch.setattr(tinydistill.teacher, "load_teacher", no_teacher)
    again = Distiller(**d.get_params()).fit(TEXTS)
    assert np.array_equal(again.teacher_scores_, d.teacher_scores_)


def test_multi_label(tmp_path):
    task = Task(
        name="toy-multi",
        labels=["price", "delivery", "quality"],
        verbalisations=["the price", "delivery", "product quality"],
        prompt="Review: {text}\nDoes the review talk about {label}? Answer yes or no.",
        multi_label=True,
    )
    gold = [["price", "quality"], None, "delivery", []] + [None] * (len(TEXTS) - 4)
    d = Distiller(task, int8=False, log=quiet, **SMALL).fit(TEXTS, gold)
    pred = d.predict(TEXTS)
    assert all(isinstance(p, list) and set(p) <= set(task.labels) for p in pred)
    assert d.predict_proba(TEXTS).shape == (len(TEXTS), 3)
    assert d.score(TEXTS, pred) == 1.0
    assert not (d.save(tmp_path / "m") / "model.int8.onnx").exists()
    assert Distiller.load(tmp_path / "m").predict(TEXTS) == pred


def test_params_and_not_fitted(task):
    d = Distiller(task, epochs=3, log=quiet)
    params = d.get_params()
    assert params["task"] is task and params["epochs"] == 3 and params["int8"] is True
    assert d.set_params(epochs=5, lr=1e-3) is d
    assert d.get_params() == params | {"epochs": 5, "lr": 1e-3}
    assert Distiller(**d.get_params()).get_params() == d.get_params()
    with pytest.raises(ValueError, match="unknown parameter"):
        d.set_params(epoch=1)
    with pytest.raises(NotFittedError, match="not fitted"):
        d.predict(["hello"])
    with pytest.raises(NotFittedError):
        d.save("nowhere")
