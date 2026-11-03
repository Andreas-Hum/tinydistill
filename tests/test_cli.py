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

    assert json.loads((run / "status.json").read_text())["finished"]
    capsys.readouterr()
    main(["status", str(run)])
    assert "eval:" in capsys.readouterr().out

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
    assert "train: 0 texts scored now" in capsys.readouterr().err

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
    assert "train: 0 texts scored now" in err  # the second run labels nothing: shared cache

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
        table[0].startswith("| Run | Teacher | Teacher accuracy | Student |") and len(table) == 6
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


def test_python_pipeline_from_the_readme(tmp_path):
    # The README's "whole pipeline" example, with tiny models.
    import torch

    from tinydistill import Predictor, export_onnx, load_task, load_teacher, read_examples, train_student

    (tmp_path / "task.yaml").write_text(TASK)
    write_jsonl(tmp_path / "d.jsonl", [{"review": f"text {i}"} for i in range(8)])
    task = load_task(tmp_path / "task.yaml")
    texts = [e.text for e in read_examples(str(tmp_path / "d.jsonl"), task, limit=6)]
    teacher = load_teacher(TINY_LM, task, device="cpu")
    soft_targets = torch.tensor(teacher.score_many(texts)).softmax(-1)
    train_student(
        texts, soft_targets, task.labels, tmp_path / "student", TINY_ENCODER, epochs=1, log=lambda m: None
    )
    export_onnx(tmp_path / "student", tmp_path / "student-onnx", int8=True)
    assert Predictor(tmp_path / "student-onnx").predict(["hello"])[0] in task.labels


def test_teacher_ensemble(tmp_path):
    (tmp_path / "task.yaml").write_text(TASK)
    rows = [
        {"review": f"it was {w}", "sentiment": "pos" if i % 2 == 0 else "neg"}
        for i, w in enumerate(["good", "bad", "great", "awful", "fine", "poor"])
    ]
    write_jsonl(tmp_path / "d.jsonl", rows)
    run = tmp_path / "run"
    mistral = "hf-internal-testing/tiny-random-MistralForCausalLM"
    main(
        [
            "run",
            str(tmp_path / "task.yaml"),
            "--data",
            str(tmp_path / "d.jsonl"),
            "--test",
            str(tmp_path / "d.jsonl"),
            "--out",
            str(run),
            "--teacher",
            f"{TINY_LM},{mistral}",
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
        ]
    )
    row = json.loads((run / "train.jsonl").read_text().splitlines()[0])
    assert len(row["teacher_scores"]) == 2
    import torch

    want = torch.tensor(row["teacher_scores"], dtype=torch.float64).softmax(-1).mean(0)
    assert torch.tensor(row["scores"], dtype=torch.float64).exp().tolist() == pytest.approx(
        want.tolist(), abs=1e-9
    )
    results = json.loads((run / "results.json").read_text())
    roles = [(m["role"], m["name"]) for m in results["models"]]
    assert roles[0] == ("teacher", "teacher ensemble of 2, zero-shot")
    assert [r for r, _ in roles].count("teacher member") == 2
    ens, members = results["models"][0], [m for m in results["models"] if m["role"] == "teacher member"]
    assert ens["latency_ms_b1"] == pytest.approx(sum(m["latency_ms_b1"] for m in members))
    assert "Speed-up is relative to the teacher ensemble" in (run / "results.md").read_text()


def test_cascade_in_eval_and_predict(tmp_path, capsys):
    (tmp_path / "task.yaml").write_text(TASK)
    rows = [
        {"review": f"it was {w}", "sentiment": "pos" if i % 2 == 0 else "neg"}
        for i, w in enumerate(["good", "bad", "great", "awful", "fine", "poor"])
    ]
    write_jsonl(tmp_path / "d.jsonl", rows)
    run = tmp_path / "run"
    main(
        [
            "run",
            str(tmp_path / "task.yaml"),
            "--data",
            str(tmp_path / "d.jsonl"),
            "--test",
            str(tmp_path / "d.jsonl"),
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
            "--int8",
        ]
    )
    results = json.loads((run / "results.json").read_text())
    assert [c["share_to_teacher"] for c in results["cascade"]][0] == 0.0
    assert results["cascade"][-1]["share_to_teacher"] == 1.0
    teacher_acc = results["models"][0]["accuracy"]
    assert results["cascade"][-1]["accuracy"] == pytest.approx(teacher_acc)  # everything to the teacher
    assert "### Cascade" in (run / "results.md").read_text()

    capsys.readouterr()
    main(
        [
            "predict",
            str(run / "onnx"),
            "good stuff",
            "bad stuff",
            "--cascade",
            str(run),
            "--threshold",
            "1.01",
            "--device",
            "cpu",
        ]
    )
    out = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [o["by"] for o in out] == ["teacher", "teacher"]  # nobody is that confident
    main(
        [
            "predict",
            str(run / "onnx"),
            "good stuff",
            "--cascade",
            str(run),
            "--threshold",
            "0",
            "--device",
            "cpu",
        ]
    )
    assert json.loads(capsys.readouterr().out)["by"] == "student"


def test_ordinal_task_reports_distances(tmp_path):
    task = TASK.replace(
        "labels:\n  pos: positive review\n  neg: negative review\n",
        "ordinal: true\nlabels:\n  low: bad review\n  mid: average review\n  high: great review\n",
    )
    (tmp_path / "task.yaml").write_text(task)
    rows = [{"review": f"review {i}", "sentiment": ["low", "mid", "high"][i % 3]} for i in range(9)]
    write_jsonl(tmp_path / "d.jsonl", rows)
    run = tmp_path / "run"
    main(
        [
            "run",
            str(tmp_path / "task.yaml"),
            "--data",
            str(tmp_path / "d.jsonl"),
            "--test",
            str(tmp_path / "d.jsonl"),
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
        ]
    )
    results = json.loads((run / "results.json").read_text())
    assert all("mae" in m and "within_one" in m for m in results["models"])
    head = (run / "results.md").read_text().splitlines()[0]
    assert "| MAE (label steps) | Within one step |" in head


MULTI_TASK = """\
name: toy-multi
text_field: review
label_field: topics
multi_label: true
prompt: |
  Review: {text}
  Does the review talk about {label}? Answer yes or no.
labels:
  price: the price
  delivery: delivery
  quality: product quality
"""


def test_multi_label_end_to_end(tmp_path, capsys):
    from tinydistill.data import read_examples
    from tinydistill.export import Cascade, Predictor
    from tinydistill.task import load_task

    (tmp_path / "task.yaml").write_text(MULTI_TASK)
    rows = [
        {
            "review": f"review {i}",
            "topics": [["price"], ["delivery", "quality"], [], ["price", "quality"]][i % 4],
        }
        for i in range(12)
    ]
    rows[0]["topics"] = "price|delivery"  # strings with | work too
    write_jsonl(tmp_path / "d.jsonl", rows)
    task = load_task(tmp_path / "task.yaml")
    assert [e.gold for e in read_examples(str(tmp_path / "d.jsonl"), task)][:3] == [[0, 1], [1, 2], []]
    run = tmp_path / "run"
    main(
        [
            "run",
            str(tmp_path / "task.yaml"),
            "--data",
            str(tmp_path / "d.jsonl"),
            "--test",
            str(tmp_path / "d.jsonl"),
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
            "--int8",
            "--gold-per-class",
            "1",
            "--gold-only",
            "--upper-bound",
        ]
    )
    results = json.loads((run / "results.json").read_text())
    assert all("micro_f1" in m for m in results["models"]) and "cascade" not in results
    assert "| Micro-F1 |" in (run / "results.md").read_text().splitlines()[0]
    pr = Predictor(run / "onnx")
    assert pr.multi_label and all(set(p) <= set(task.labels) for p in pr.predict(["a", "b"]))
    with pytest.raises(ValueError, match="single-label"):
        Cascade(pr, None, 0.5)
    capsys.readouterr()
    main(["predict", str(run / "onnx"), "great price"])
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"text", "labels", "scores"} and set(out["scores"]) == set(task.labels)


def test_synth_command_runs(tmp_path):
    (tmp_path / "task.yaml").write_text(TASK)
    out = tmp_path / "synth.jsonl"
    main(
        [
            "synth",
            str(tmp_path / "task.yaml"),
            "--per-label",
            "2",
            "--out",
            str(out),
            "--teacher",
            TINY_LM,
            "--device",
            "cpu",
            "--temperature",
            "1.0",
        ]
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]  # a random tiny model writes gibberish
    assert all(set(r) == {"review", "prompted_label"} for r in rows)


def test_report_rebuilds_the_markdown(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    results = {
        "task": "t",
        "teacher": "T",
        "student": "S",
        "data": {"n_train": 2, "n_test": 2},
        "environment": {"cpu_threads": 2},
        "models": [
            {
                "role": "teacher",
                "name": "teacher, zero-shot",
                "device": "cpu",
                "accuracy": 0.5,
                "latency_ms_b1": 1000.0,
            },
            {
                "role": "teacher",
                "name": "teacher, zero-shot",
                "device": "cuda",
                "accuracy": 0.5,
                "latency_ms_b1": 50.0,
            },
            {
                "role": "student",
                "name": "student, ONNX int8",
                "device": "cpu",
                "accuracy": 0.4,
                "latency_ms_b1": 2.0,
            },
        ],
        "cascade": [{"share_to_teacher": 0.5, "threshold": 0.6, "accuracy": 0.45, "macro_f1": 0.4}],
    }
    (run / "results.json").write_text(json.dumps(results))
    main(["report", str(run)])
    md = (run / "results.md").read_text()
    # the cascade uses the fastest teacher: 2 + 0.5 * 50 = 27 ms
    assert "| 50% | 0.600 | 0.450 | 0.400 | 27.0 |" in md and "teacher, zero-shot on cuda." in md
