import json

from conftest import TINY_ENCODER, TINY_LM

from tinydistill.cli import main

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


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_end_to_end(tmp_path, capsys):
    (tmp_path / "task.yaml").write_text(TASK)
    words = ["good", "bad", "great", "awful", "fine", "terrible"]
    write_jsonl(tmp_path / "train.jsonl", [{"review": f"this was {w} {i}"} for i, w in enumerate(words * 3)])
    write_jsonl(
        tmp_path / "test.csv.jsonl",
        [{"review": f"so {w}", "sentiment": "pos" if i % 2 == 0 else "neg"} for i, w in enumerate(words)],
    )
    run = tmp_path / "run"
    main(
        [
            "run",
            str(tmp_path / "task.yaml"),
            "--data",
            str(tmp_path / "train.jsonl"),
            "--test",
            str(tmp_path / "test.csv.jsonl"),
            "--out",
            str(run),
            "--teacher",
            TINY_LM,
            "--student",
            TINY_ENCODER,
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--int8",
            "--reps",
            "2",
            "--teacher-reps",
            "2",
            "--device",
            "cpu",
            "--threads",
            "2",
        ]
    )
    results = json.loads((run / "results.json").read_text())
    names = [m["name"] for m in results["models"]]
    assert names == ["teacher, zero-shot", "student, PyTorch", "student, ONNX fp32", "student, ONNX int8"]
    for m in results["models"]:
        assert 0 <= m["accuracy"] <= 1 and m["latency_ms_b1"] > 0
    assert results["data"] == {
        "train": str(tmp_path / "train.jsonl"),
        "test": str(tmp_path / "test.csv.jsonl"),
        "n_train": 18,
        "n_test": 6,
    }
    md = (run / "results.md").read_text()
    assert md.startswith("| Model | Device |") and "student, ONNX int8" in md

    # Labelling again resumes entirely from the cache.
    capsys.readouterr()
    main(
        [
            "label",
            str(tmp_path / "task.yaml"),
            "--data",
            str(tmp_path / "train.jsonl"),
            "--out",
            str(run),
            "--teacher",
            TINY_LM,
            "--device",
            "cpu",
        ]
    )
    assert "train: 18 texts (0 scored now" in capsys.readouterr().err

    main(["predict", str(run / "onnx"), "what a great day", "--fp32"])
    out = json.loads(capsys.readouterr().out.strip())
    assert out["label"] in ("pos", "neg") and 0 < out["confidence"] <= 1
