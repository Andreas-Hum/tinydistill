"""The task file's `outside` label: texts that fit none of the other labels."""

import json
from pathlib import Path

import pytest
import yaml
from conftest import TINY_ENCODER, TINY_LM

from tinydistill.cli import main
from tinydistill.evaluate import outside_metrics
from tinydistill.task import load_task

TASK = """\
name: toy
text_field: review
label_field: sentiment
prompt: |
  Classify the sentiment of the review as one of:
  {labels}

  Review: {text}
labels:
  pos: positive review
  neg: negative review
"""


def write(tmp_path, extra: str = "outside: true\n") -> Path:
    (tmp_path / "task.yaml").write_text(TASK + extra)
    return tmp_path / "task.yaml"


def test_outside_is_the_last_label_and_round_trips(tmp_path):
    task = load_task(write(tmp_path))
    assert task.outside and task.outside_id == "outside"
    assert task.labels == ["pos", "neg", "outside"]
    assert task.verbalisations == ["positive review", "negative review", "none of these"]
    assert task.render("ok").endswith(
        "- negative review\n- none of these: the text fits none of the other categories\n\nReview: ok\n"
    )
    custom = load_task(write(tmp_path, "outside: {id: oos, name: something else, description: ''}\n"))
    assert custom.labels[-1] == "oos" and custom.verbalisations[-1] == "something else"
    assert custom.descriptions == []  # an empty description leaves it out
    for t in (task, custom):  # the run directory stores the task via to_dict()
        (tmp_path / "saved.yaml").write_text(yaml.safe_dump(t.to_dict(), sort_keys=False))
        assert load_task(tmp_path / "saved.yaml") == t
    assert not load_task(write(tmp_path, "")).outside


@pytest.mark.parametrize(
    "extra, message",
    [
        ("outside: true\nmulti_label: true\n", "single-label"),
        ("outside: {id: pos}\n", "also one of the labels"),
        ("outside: {name: negative review}\n", "unique"),
        ("outside: {label: x}\n", "true or a mapping"),
        ("outside: yes please\n", "true or a mapping"),
    ],
)
def test_bad_outside_keys_are_refused(tmp_path, extra, message):
    with pytest.raises(ValueError, match=message):
        load_task(write(tmp_path, extra))


def test_outside_metrics_on_hand_made_predictions():
    gold = [0, 2, 1, 1, 0]
    m = outside_metrics([0, 2, 2, 1, 2], gold, outside=2)
    assert m == {"in_scope_accuracy": 0.5, "outside_recall": 1.0, "outside_precision": pytest.approx(1 / 3)}
    assert outside_metrics([0, 1, 0, 1, 0], gold, 2) == {
        "in_scope_accuracy": 0.75,
        "outside_recall": 0.0,
        "outside_precision": 0.0,  # never predicted: 0, as macro_f1 counts a class without true positives
    }
    assert outside_metrics([0, 2], [0, 1], 2) == {}  # no gold `outside` label: nothing to report


def test_outside_through_a_run(tmp_path, capsys):
    words = ["good", "bad", "great", "awful", "fine", "terrible"]
    (tmp_path / "train.jsonl").write_text(
        "".join(json.dumps({"review": f"this was {w} {i}"}) + "\n" for i, w in enumerate(words * 2))
    )
    gold = ["pos", "neg", "outside"] * 2
    rows = [{"review": f"so {w}", "sentiment": g} for w, g in zip(words, gold, strict=True)]
    (tmp_path / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    run = tmp_path / "run"
    main(
        ["run", str(write(tmp_path)), "--data", str(tmp_path / "train.jsonl"), "--test"]
        + [str(tmp_path / "test.jsonl"), "--out", str(run), "--teacher", TINY_LM, "--student", TINY_ENCODER]
        + ["--epochs", "1", "--batch-size", "4", "--reps", "2", "--skip-teacher-bench", "--device", "cpu"]
    )
    rows = [json.loads(line) for line in (run / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["gold"] for r in rows] == [0, 1, 2, 0, 1, 2] and all(len(r["scores"]) == 3 for r in rows)
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    for m in results["models"]:
        assert {"in_scope_accuracy", "outside_recall", "outside_precision"} <= set(m)
    md = (run / "results.md").read_text(encoding="utf-8")
    assert "| In-scope accuracy | Out-of-scope recall | Out-of-scope precision |" in md

    from tinydistill.export import Predictor

    pr = Predictor(run / "onnx")
    assert pr.outside == "outside"
    for label, d in zip(pr.predict(words), pr.describe(words), strict=True):
        assert d["label"] == label and d["outside"] == (label == "outside")
    capsys.readouterr()
    main(["predict", str(run / "onnx"), "what a day"])
    out = json.loads(capsys.readouterr().out)
    assert out["outside"] == (out["label"] == "outside")

    from tinydistill.web import build_site

    page = build_site(run / "onnx", tmp_path / "site").read_text(encoding="utf-8")
    data = json.loads(page.split('id="data">')[1].split("</script>")[0].replace("<\\/", "</"))
    assert data["outside"] == 2 and '"None of the labels fits"' in page


def test_the_explainer_answers_in_words_for_the_outside_label():
    import tinydistill

    page = (Path(tinydistill.__file__).parent / "explainer.html").read_text(encoding="utf-8")
    assert 'i === D.outside ? "None of the labels fits"' in page
