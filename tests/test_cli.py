import json

import pytest
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


def test_few_labels_upper_bound_and_shared_cache(tmp_path, capsys):
    (tmp_path / "task.yaml").write_text(TASK)
    words = ["good", "bad", "great", "awful", "fine", "terrible"]
    rows = [
        {"review": f"it was {w} {i}", "sentiment": "pos" if j % 2 == 0 else "neg"}
        for i in range(3)
        for j, w in enumerate(words)
    ]
    write_jsonl(tmp_path / "train.jsonl", rows)
    write_jsonl(tmp_path / "test.jsonl", rows[:6])
    common = [
        str(tmp_path / "task.yaml"),
        "--data",
        str(tmp_path / "train.jsonl"),
        "--test",
        str(tmp_path / "test.jsonl"),
        "--teacher",
        TINY_LM,
        "--student",
        TINY_ENCODER,
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--reps",
        "2",
        "--teacher-reps",
        "2",
        "--device",
        "cpu",
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    main(["run", *common, "--out", str(tmp_path / "a")])
    capsys.readouterr()
    main(
        [
            "run",
            *common,
            "--out",
            str(tmp_path / "b"),
            "--gold-per-class",
            "2",
            "--upper-bound",
            "--gold-only",
            "--seeds",
            "2",
        ]
    )
    err = capsys.readouterr().err
    assert "train: 18 texts (0 scored now" in err  # the second run labels nothing: shared cache

    results = json.loads((tmp_path / "b" / "results.json").read_text())
    assert results["training"]["gold_per_class"] == 2
    assert results["training"]["n_gold"] == 4  # 2 per class, 2 classes
    assert results["training"]["n_gold_upper_bound"] == 18
    upper = [m for m in results["models"] if m["role"] == "upper bound"]
    assert len(upper) == 1 and 0 <= upper[0]["accuracy"] <= 1
    md = (tmp_path / "b" / "results.md").read_text()
    assert "cross-entropy on 2 gold labels per class (4 texts" in md and "upper bound" in md
    roles = [m["role"] for m in results["models"]]
    assert roles.count("baseline") == 1 and roles.count("student") == 2  # PyTorch row + ONNX fp32 row
    student = results["models"][roles.index("student")]
    assert len(student["accuracy_seeds"]) == 2 and "accuracy_std" in student
    assert (tmp_path / "b" / "seeds" / "1" / "student-kgold").is_dir()
    assert " ± " in md and "over 2 training seeds" in md and "on 2 gold labels per class, no teacher" in md
    capsys.readouterr()
    main(["compare", str(tmp_path / "a"), str(tmp_path / "b" / "results.json")])
    table = capsys.readouterr().out.splitlines()
    assert (
        table[0].startswith("| Run | Teacher | Student |") and len(table) == 6
    )  # header, rule, 2 rows, note
    assert table[2].startswith("| a |") and table[3].startswith("| b/results |") and " ± " in table[3]
    first = json.loads((tmp_path / "a" / "results.json").read_text())
    assert first["training"]["gold_per_class"] == 0 and not first["training"]["upper_bound"]


def test_few_shot_teacher(tmp_path, capsys):
    task = TASK.replace("  {labels}\n", "  {labels}\n  {examples}\n")
    (tmp_path / "task.yaml").write_text(task)
    rows = [
        {"review": f"it was {w} {i}", "sentiment": "pos" if j % 2 == 0 else "neg"}
        for i in range(3)
        for j, w in enumerate(["good", "bad", "great", "awful"])
    ]
    write_jsonl(tmp_path / "train.jsonl", rows)
    write_jsonl(tmp_path / "test.jsonl", rows[:4])
    run = tmp_path / "run"
    main(
        [
            "run",
            str(tmp_path / "task.yaml"),
            "--data",
            str(tmp_path / "train.jsonl"),
            "--test",
            str(tmp_path / "test.jsonl"),
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
            "--reps",
            "2",
            "--teacher-reps",
            "2",
            "--device",
            "cpu",
            "--few-shot",
            "1",
            "--gold-per-class",
            "1",
        ]
    )
    examples = [json.loads(line) for line in (run / "examples.jsonl").read_text().splitlines()]
    assert len(examples) == 2 and {e["gold"] for e in examples} == {0, 1}
    results = json.loads((run / "results.json").read_text())
    assert results["models"][0]["name"] == "teacher, 1-shot"
    assert results["training"]["few_shot"] == 1 and results["training"]["n_examples"] == 2
    assert "prompt shows 1 gold training examples per class (2 texts)" in (run / "results.md").read_text()

    (tmp_path / "plain.yaml").write_text(TASK)  # no {examples} line
    with pytest.raises(SystemExit, match="examples"):
        main(
            [
                "label",
                str(tmp_path / "plain.yaml"),
                "--data",
                str(tmp_path / "train.jsonl"),
                "--out",
                str(tmp_path / "x"),
                "--teacher",
                TINY_LM,
                "--device",
                "cpu",
                "--few-shot",
                "1",
            ]
        )


def test_init_writes_a_loadable_task(tmp_path):
    from tinydistill.task import load_task

    write_jsonl(tmp_path / "d.jsonl", [{"body": "great", "tag": "pos"}, {"body": "awful", "tag": "neg"}])
    out = tmp_path / "task.yaml"
    main(
        ["init", str(tmp_path / "d.jsonl"), "--out", str(out), "--text-field", "body", "--label-field", "tag"]
    )
    task = load_task(out)
    assert task.labels == ["neg", "pos"] and task.text_field == "body" and "{examples}" in task.prompt
    with pytest.raises(SystemExit, match="exists"):
        main(["init", str(tmp_path / "d.jsonl"), "--out", str(out)])
    write_jsonl(tmp_path / "n.jsonl", [{"text": "a", "label": 0}, {"text": "b", "label": 1}])
    with pytest.raises(ValueError, match="numbers without names"):
        main(["init", str(tmp_path / "n.jsonl"), "--out", str(tmp_path / "n.yaml")])


def test_push_writes_a_model_card_and_uploads(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []

    class FakeApi:
        def create_repo(self, repo, private, exist_ok):
            calls.append(("create", repo, private))
            return f"https://huggingface.co/{repo}"

        def upload_folder(self, folder_path, repo_id, commit_message):
            calls.append(("upload", folder_path, repo_id))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    onnx = tmp_path / "run" / "onnx"
    onnx.mkdir(parents=True)
    (onnx / "labels.json").write_text('["neg", "pos"]')
    (tmp_path / "run" / "meta.json").write_text('{"teacher": "T", "student": "S"}')
    (tmp_path / "run" / "results.md").write_text("| Model |\n|---|\n| x |\n")
    main(["push", str(onnx), "--repo", "me/clf", "--private", "--license", "apache-2.0"])
    card = (onnx / "README.md").read_text()
    assert "license: apache-2.0" in card and "`T`" in card and "## Results" in card and '"me/clf"' in card
    assert calls == [("create", "me/clf", True), ("upload", str(onnx), "me/clf")]


def test_python_api_exports():
    import tinydistill

    for name in tinydistill.__all__:
        assert hasattr(tinydistill, name)
