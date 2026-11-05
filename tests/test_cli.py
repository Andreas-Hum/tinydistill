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
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
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
    md = (run / "results.md").read_text(encoding="utf-8")
    assert md.startswith("| Model | Device |") and "student, ONNX int8" in md

    assert json.loads((run / "status.json").read_text(encoding="utf-8"))["finished"]
    capsys.readouterr()
    main(["status", str(run)])
    assert "eval:" in capsys.readouterr().out

    main(["analyse", str(run)])
    out = capsys.readouterr().out
    assert out.startswith("Teacher accuracy") and (run / "analysis.html").read_text(
        encoding="utf-8"
    ).startswith("<!doctype html>")

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
    assert "the teacher is not loaded" in capsys.readouterr().err  # resumed: everything is cached

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
    assert "every text is in the cache, so the teacher is not loaded" in err  # shared cache

    results = json.loads((tmp_path / "b" / "results.json").read_text(encoding="utf-8"))
    assert results["training"]["gold_per_class"] == 2
    assert results["training"]["n_gold"] == 4  # 2 per class, 2 classes
    assert results["training"]["n_gold_upper_bound"] == 18
    upper = [m for m in results["models"] if m["role"] == "upper bound"]
    assert len(upper) == 1 and 0 <= upper[0]["accuracy"] <= 1
    md = (tmp_path / "b" / "results.md").read_text(encoding="utf-8")
    assert "cross-entropy on 4 gold-labelled texts (2 per class; weight 1.0)" in md and "upper bound" in md
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
    first = json.loads((tmp_path / "a" / "results.json").read_text(encoding="utf-8"))
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
    examples = [
        json.loads(line) for line in (run / "examples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(examples) == 2 and {e["gold"] for e in examples} == {0, 1}
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert results["models"][0]["name"] == "teacher, 1-shot"
    assert results["training"]["few_shot"] == 1 and results["training"]["n_examples"] == 2
    assert "prompt shows 1 gold training examples per class (2 texts)" in (run / "results.md").read_text(
        encoding="utf-8"
    )


def test_few_shot_similar_teacher(tmp_path, capsys):
    from tinydistill.cli import _run_teacher

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
        ["run", str(tmp_path / "task.yaml"), "--data", str(tmp_path / "train.jsonl")]
        + ["--test", str(tmp_path / "test.jsonl"), "--out", str(run), "--teacher", TINY_LM]
        + ["--student", TINY_ENCODER, "--epochs", "1", "--batch-size", "4", "--device", "cpu"]
        + ["--reps", "2", "--teacher-reps", "2", "--few-shot-similar", "2", "--few-shot-model", TINY_ENCODER]
    )
    examples = [
        json.loads(line) for line in (run / "examples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [e["text"] for e in examples] == [r["review"] for r in rows]  # the pool: every gold training text
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    assert meta["few_shot_similar"] == 2 and meta["few_shot_model"] == TINY_ENCODER
    assert _run_teacher(run, "cpu").fingerprint() == meta["fingerprint"]  # the cascade's teacher is the same
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert results["models"][0]["name"] == "teacher, 2 similar examples"
    assert "each text the 2 most similar of 12 gold training examples" in (run / "results.md").read_text(
        encoding="utf-8"
    )
    (tmp_path / "after.yaml").write_text(
        task.replace("  {examples}\n", "").replace("{text}", "{text}\n  {examples}")
    )
    with pytest.raises(SystemExit, match="before"):
        main(
            ["label", str(tmp_path / "after.yaml"), "--data", str(tmp_path / "train.jsonl")]
            + ["--teacher", TINY_LM, "--few-shot-similar", "2", "--out", str(tmp_path / "x")]
        )
    capsys.readouterr()

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
    assert "add `outside: true`" in out.read_text(encoding="utf-8")  # offered, not added
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
    (tmp_path / "run" / "meta.json").write_text(
        '{"teacher": "T", "student": "org/S", "data": "hf:org/ds:train", "test": "hf:org/ds:test"}'
    )
    (tmp_path / "run" / "results.md").write_text("| Model |\n|---|\n| x |\n")
    main(["push", str(onnx), "--repo", "me/clf", "--private", "--license", "apache-2.0"])
    card = (onnx / "README.md").read_text(encoding="utf-8")
    assert "license: apache-2.0" in card and "`T`" in card and "## Results" in card and '"hf:me/clf"' in card
    assert "base_model: org/S" in card and "datasets:\n- org/ds\n---" in card
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
    row = json.loads((run / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert len(row["teacher_scores"]) == 2
    import torch

    want = torch.tensor(row["teacher_scores"], dtype=torch.float64).softmax(-1).mean(0)
    assert torch.tensor(row["scores"], dtype=torch.float64).exp().tolist() == pytest.approx(
        want.tolist(), abs=1e-9
    )
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    roles = [(m["role"], m["name"]) for m in results["models"]]
    assert roles[0] == ("teacher", "teacher ensemble of 2, zero-shot")
    assert [r for r, _ in roles].count("teacher member") == 2
    ens, members = results["models"][0], [m for m in results["models"] if m["role"] == "teacher member"]
    assert ens["latency_ms_b1"] == pytest.approx(sum(m["latency_ms_b1"] for m in members))
    assert "Speed-up is relative to the teacher ensemble" in (run / "results.md").read_text(encoding="utf-8")


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
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert [c["share_to_teacher"] for c in results["cascade"]][0] == 0.0
    assert results["cascade"][-1]["share_to_teacher"] == 1.0
    teacher_acc = results["models"][0]["accuracy"]
    assert results["cascade"][-1]["accuracy"] == pytest.approx(teacher_acc)  # everything to the teacher
    assert "### Cascade" in (run / "results.md").read_text(encoding="utf-8")

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
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert all("mae" in m and "within_one" in m for m in results["models"])
    head = (run / "results.md").read_text(encoding="utf-8").splitlines()[0]
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
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert all("micro_f1" in m for m in results["models"]) and "cascade" not in results
    assert "| Micro-F1 |" in (run / "results.md").read_text(encoding="utf-8").splitlines()[0]
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
    rows = [
        json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
    ]  # a random tiny model writes gibberish
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
    md = (run / "results.md").read_text(encoding="utf-8")
    # the cascade uses the fastest teacher: 2 + 0.5 * 50 = 27 ms
    assert "| 50% | 0.600 | 0.450 | 0.400 | 27.0 |" in md and "teacher, zero-shot on cuda." in md


def test_suggest_then_train_with_the_gold_file(tmp_path, capsys):
    (tmp_path / "task.yaml").write_text(TASK)
    rows = [{"review": f"text {i}"} for i in range(12)]  # no gold labels at all
    write_jsonl(tmp_path / "d.jsonl", rows)
    run = tmp_path / "run"
    common = [
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
            *common,
        ]
    )
    out = tmp_path / "to_label.jsonl"
    main(["suggest", str(run), "--n", "4", "--out", str(out)])
    picked = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert (
        len(picked) == 4
        and len({r["review"] for r in picked}) == 4
        and all(r["sentiment"] == "" for r in picked)
    )
    for i, r in enumerate(picked):  # a person labels them
        r["sentiment"] = "pos" if i % 2 else "neg"
    write_jsonl(out, picked)
    capsys.readouterr()
    main(["train", str(run), "--gold-file", str(out), "--gold-only", *common[2:6], "--device", "cpu"])
    assert "--gold-file: 4 of 4 labelled texts are in the training data" in capsys.readouterr().err
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_gold_file"] == 4 and meta["n_gold"] == 4 and (run / "student-kgold").is_dir()


def test_tune_tries_every_setting_on_a_dev_split(tmp_path, capsys):
    (tmp_path / "task.yaml").write_text(TASK.replace("  {labels}\n", "  {labels}\n  {examples}\n"))
    (tmp_path / "alt.yaml").write_text(TASK.replace("Classify the sentiment", "What is the sentiment"))
    rows = [
        {"review": f"it was {w} {i}", "sentiment": "pos" if j % 2 == 0 else "neg"}
        for i in range(4)
        for j, w in enumerate(["good", "bad", "great", "awful"])
    ]
    write_jsonl(tmp_path / "d.jsonl", rows)
    out = tmp_path / "tune-1.5b"  # a dot in the name: the files are tune-1.5b.md and tune-1.5b.json
    main(
        [
            "tune",
            str(tmp_path / "task.yaml"),
            "--prompt",
            str(tmp_path / "alt.yaml"),
            "--data",
            str(tmp_path / "d.jsonl"),
            "--teacher",
            TINY_LM,
            "--few-shot",
            "0,1",
            "--dev-size",
            "6",
            "--device",
            "cpu",
            "--out",
            str(out),
        ]
    )
    results = json.loads((tmp_path / "tune-1.5b.json").read_text(encoding="utf-8"))
    # task.yaml with 0 and 1 examples; alt.yaml only zero-shot (it has no {examples} line)
    assert sorted((r["prompt"], r["few_shot"]) for r in results) == [
        ("alt.yaml", 0),
        ("task.yaml", 0),
        ("task.yaml", 1),
    ]
    assert [r["accuracy"] for r in results] == sorted((r["accuracy"] for r in results), reverse=True)
    assert "Best: `tinydistill run" in (tmp_path / "tune-1.5b.md").read_text(encoding="utf-8")


def test_tune_needs_a_local_teacher(monkeypatch):
    import tinydistill.tune

    monkeypatch.setattr(tinydistill.tune, "load_teacher", lambda *a, **k: object())  # not a LabelScorer
    with pytest.raises(SystemExit, match="needs a local teacher"):
        tinydistill.tune.tune([("t", None)], ["a", "b"], [0, 1], ["m"], [0], 2, "cpu", False, 0, print)


def test_cascade_log_then_refresh(tmp_path, capsys):
    (tmp_path / "task.yaml").write_text(TASK)
    write_jsonl(
        tmp_path / "d.jsonl",
        [{"review": f"text {i}", "sentiment": "pos" if i % 2 else "neg"} for i in range(8)],
    )
    run, new = tmp_path / "run", tmp_path / "run2"
    common = [
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
            *common,
        ]
    )
    log_file = tmp_path / "cascade.jsonl"
    main(
        [
            "predict",
            str(run / "onnx"),
            "brand new text",
            "another new one",
            "--cascade",
            str(run),
            "--threshold",
            "1.01",
            "--log",
            str(log_file),
            "--device",
            "cpu",
        ]
    )
    logged = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines()]
    assert [r["text"] for r in logged] == ["brand new text", "another new one"] and len(
        logged[0]["scores"]
    ) == 2
    capsys.readouterr()
    main(["refresh", str(run), "--log", str(log_file), "--out", str(new), "--skip-teacher-bench", *common])
    assert "plus 2 the teacher answered" in capsys.readouterr().err
    assert (
        len((new / "train.jsonl").read_text(encoding="utf-8").splitlines()) == 10
        and (new / "results.json").exists()
    )
    assert json.loads((new / "meta.json").read_text(encoding="utf-8"))["refresh_added"] == 2
    status = json.loads((new / "status.json").read_text(encoding="utf-8"))
    assert status["stage"] == "eval" and status["finished"] and status["done"] == status["total"]


def test_shortlist_scores_k_labels_and_reports_recall(tmp_path):
    from tinydistill.teacher import NOT_SCORED

    (tmp_path / "task.yaml").write_text(TASK)
    rows = [
        {"review": f"it was {w} {i}", "sentiment": s}
        for i in range(3)
        for w, s in [("good", "pos"), ("bad", "neg")]
    ]
    write_jsonl(tmp_path / "train.jsonl", rows)
    write_jsonl(tmp_path / "test.jsonl", rows)
    run = tmp_path / "run"
    main(
        [
            "run",
            str(tmp_path / "task.yaml"),
            "--data",
            str(tmp_path / "train.jsonl"),
            "--test",
            str(tmp_path / "test.jsonl"),
            "--teacher",
            TINY_LM,
            "--student",
            TINY_ENCODER,
            "--shortlist",
            "1",
            "--shortlist-model",
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
            "--out",
            str(run),
        ]
    )
    test = [json.loads(line) for line in (run / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(sum(x > NOT_SCORED / 2 for x in r["scores"]) == 1 for r in test)
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    want = sum(r["scores"][r["gold"]] > NOT_SCORED / 2 for r in test) / len(test)
    assert results["training"]["shortlist_recall"] == want
    assert "the 1 labels most similar to each text" in (run / "results.md").read_text(encoding="utf-8")
    kept = tmp_path / "kept"
    main(
        ["label", str(tmp_path / "task.yaml"), "--data", str(tmp_path / "train.jsonl"), "--teacher", TINY_LM]
        + ["--shortlist", "1", "--shortlist-model", TINY_ENCODER, "--shortlist-keep", "neg"]
        + ["--device", "cpu", "--out", str(kept)]
    )
    rows = [json.loads(line) for line in (kept / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["scores"][1] > NOT_SCORED / 2 for r in rows)  # neg is always scored
    assert json.loads((kept / "meta.json").read_text(encoding="utf-8"))["shortlist_keep"] == ["neg"]


def test_label_skips_loading_the_teacher_when_everything_is_cached(tmp_path, monkeypatch, capsys):
    import tinydistill.teacher

    (tmp_path / "task.yaml").write_text(TASK)
    rows = [
        {"review": f"it was {w}", "sentiment": s} for w, s in [("good", "pos"), ("bad", "neg"), ("ok", "pos")]
    ]
    write_jsonl(tmp_path / "d.jsonl", rows)
    args = ["label", str(tmp_path / "task.yaml"), "--data", str(tmp_path / "d.jsonl"), "--teacher", TINY_LM]
    args += ["--device", "cpu", "--cache-dir", str(tmp_path / "cache")]
    main([*args, "--limit", "2", "--out", str(tmp_path / "a")])
    real = tinydistill.teacher.load_teacher

    def no_loading(*_, **__):
        raise AssertionError("the teacher was loaded")

    monkeypatch.setattr(tinydistill.teacher, "load_teacher", no_loading)
    main([*args, "--limit", "2", "--out", str(tmp_path / "b")])
    assert (tmp_path / "a" / "train.jsonl").read_text(encoding="utf-8") == (
        tmp_path / "b" / "train.jsonl"
    ).read_text(encoding="utf-8")
    assert "not loaded" in capsys.readouterr().err
    with pytest.raises(AssertionError, match="was loaded"):  # a new text needs the teacher
        main([*args, "--out", str(tmp_path / "c")])
    monkeypatch.setattr(tinydistill.teacher, "load_teacher", real)
    main([*args, "--out", str(tmp_path / "c")])
    assert "1 texts scored now" in capsys.readouterr().err


def test_models_load_from_the_hub(tmp_path, monkeypatch, capsys):
    import huggingface_hub
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from tinydistill import Predictor
    from tinydistill.export import export_onnx

    model = AutoModelForSequenceClassification.from_pretrained(
        TINY_ENCODER, num_labels=2, id2label={0: "neg", 1: "pos"}, ignore_mismatched_sizes=True
    )
    model.save_pretrained(tmp_path / "student")
    AutoTokenizer.from_pretrained(TINY_ENCODER).save_pretrained(tmp_path / "student")
    export_onnx(tmp_path / "student", tmp_path / "onnx")
    asked = []

    def fake_download(repo, allow_patterns):
        asked.append(repo)
        return str(tmp_path / "onnx")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    assert Predictor("hf:me/clf").predict(["good"])[0] in ("neg", "pos")
    main(["predict", "hf:me/clf", "good"])
    assert json.loads(capsys.readouterr().out)["label"] in ("neg", "pos")
    assert asked == ["me/clf", "me/clf"]
    with pytest.raises(FileNotFoundError, match="hf:user/name"):
        Predictor(tmp_path / "missing")


def test_train_keeps_the_best_epoch_on_held_out_texts(tmp_path):
    (tmp_path / "task.yaml").write_text(TASK)
    rows = [
        {"review": f"it was {w} {i}", "sentiment": s}
        for i in range(5)
        for w, s in [("good", "pos"), ("bad", "neg")]
    ]
    write_jsonl(tmp_path / "d.jsonl", rows)
    run = tmp_path / "run"
    main(
        [
            "label",
            str(tmp_path / "task.yaml"),
            "--data",
            str(tmp_path / "d.jsonl"),
            "--test",
            str(tmp_path / "d.jsonl"),
            "--teacher",
            TINY_LM,
            "--device",
            "cpu",
            "--out",
            str(run),
        ]
    )
    main(
        [
            "train",
            str(run),
            "--student",
            TINY_ENCODER,
            "--epochs",
            "3",
            "--val-share",
            "0.2",
            "--patience",
            "1",
            "--gold-per-class",
            "1",
            "--gold-only",
            "--init-head",
            "names",
            "--device",
            "cpu",
        ]
    )
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    assert meta["val_share"] == 0.2 and set(meta["kept_epochs"]) == {"student", "student-kgold"}
    assert meta["init_head"] == "names"
    assert all(1 <= e <= 3 for runs in meta["kept_epochs"].values() for e in runs)
    main(["export", str(run)])
    main(["eval", str(run), "--reps", "2", "--teacher-reps", "2", "--skip-teacher-bench"])
    assert "holds out 20% of its training texts and keeps the epoch" in (run / "results.md").read_text(
        encoding="utf-8"
    )

    history = json.loads((run / "student" / "history.json").read_text(encoding="utf-8"))
    assert (
        history["kept_epoch"] == meta["kept_epochs"]["student"][0] and "held_out_loss" in history["epochs"][0]
    )
    main(["plot", str(run)])
    assert {p.name for p in (run / "plots").glob("*.png")} == {
        "training.png",
        "speed_accuracy.png",
        "per_label.png",
        "confusions.png",
        "cascade.png",
        "soft_labels.png",
    }


def test_env_lists_versions_and_survives_missing_packages(monkeypatch, capsys):
    import importlib.metadata as md

    real = md.version

    def fake(pkg):
        if pkg == "bitsandbytes":
            raise md.PackageNotFoundError(pkg)
        return real(pkg)

    monkeypatch.setattr(md, "version", fake)
    main(["env"])
    out = capsys.readouterr().out
    for key in (
        "tinydistill",
        "Python",
        "Platform",
        "torch",
        "CUDA available",
        "transformers",
        "onnxruntime",
    ):
        assert f"- {key}: " in out
    assert "- bitsandbytes: not installed" in out


def test_budget_rounds_label_the_texts_the_student_is_least_sure_of(tmp_path, monkeypatch):
    import torch

    from tinydistill import student
    from tinydistill.suggest import uncertainty

    uniform = torch.zeros(1, 4)
    assert uncertainty(uniform).item() == pytest.approx(torch.log(torch.tensor(4.0)).item())
    assert uncertainty(uniform, multi_label=True).item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    assert uncertainty(torch.tensor([[0.0, 1.0], [0.0, 5.0]])).argmax().item() == 0

    asked: list[list[str]] = []

    def fake_logits(model, tok, texts, **kw):  # text i: logits [0, i], so the lower i the less sure
        asked.append(list(texts))
        return torch.tensor([[0.0, float(t.split()[-1])] for t in texts])

    monkeypatch.setattr(student, "student_logits", fake_logits)
    (tmp_path / "task.yaml").write_text(TASK)
    write_jsonl(tmp_path / "d.jsonl", [{"review": f"text {i}"} for i in range(12)])
    run = tmp_path / "run"
    main(
        [
            "run",
            str(tmp_path / "task.yaml"),
            "--data",
            str(tmp_path / "d.jsonl"),
            "--out",
            str(run),
            "--budget",
            "7",
            "--rounds",
            "3",
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
            "--skip-teacher-bench",
            "--device",
            "cpu",
        ]
    )

    def by_i(texts):
        return sorted(texts, key=lambda t: int(t.split()[-1]))

    # 7 texts in 3 rounds: 2 at random, then the 2 and the 3 the student is least sure of (eval's timing
    # calls come after these two)
    first = {f"text {i}" for i in range(12)} - set(asked[0])
    second, third = set(by_i(asked[0])[:2]), set(by_i(asked[1])[:3])
    assert len(first) == 2 and len(asked[0]) == 10 and set(asked[1]) == set(asked[0]) - second
    train = {r["text"] for r in map(json.loads, (run / "train.jsonl").read_text().splitlines())}
    assert train == first | second | third and len(train) == 7
    (cache,) = run.glob("teacher-*.jsonl")
    assert len(cache.read_text().splitlines()) == 7  # the teacher labelled exactly the budget
    meta = json.loads((run / "meta.json").read_text())
    assert meta["budget"] == 7 and meta["rounds"] == 3
    assert json.loads((run / "results.json").read_text())["training"]["budget"] == 7


def test_analyse_review_lists_gold_contradictions_then_divided_ensembles(tmp_path, capsys):
    import csv
    import math

    run = tmp_path / "run"
    run.mkdir()
    (run / "task.yaml").write_text(TASK + "  neu: neutral review\n")

    def lg(*p):  # log-probabilities: scores whose softmax is p
        return [math.log(x) for x in p]

    a, b, c = lg(0.8, 0.1, 0.1), lg(0.1, 0.8, 0.1), lg(0.1, 0.1, 0.8)
    rows = [
        {"text": "agreed", "gold": 0, "scores": lg(0.7, 0.2, 0.1), "teacher_scores": [a, a, a]},
        {"text": "unsure contradiction", "gold": 0, "scores": lg(0.3, 0.6, 0.1), "teacher_scores": [b, b, a]},
        {"text": "split three ways", "gold": None, "scores": lg(0.4, 0.3, 0.3), "teacher_scores": [a, b, c]},
        {"text": "sure contradiction", "gold": 2, "scores": lg(0.05, 0.9, 0.05), "teacher_scores": [b, b, b]},
        {"text": "split two to one", "gold": 1, "scores": lg(0.2, 0.7, 0.1), "teacher_scores": [b, b, c]},
    ]
    write_jsonl(run / "train.jsonl", rows)
    out = tmp_path / "review.csv"
    main(["analyse", str(run), "--review", str(out)])
    with out.open(encoding="utf-8", newline="") as f:
        got = list(csv.DictReader(f))
    assert [r["text"] for r in got] == [
        "sure contradiction",
        "unsure contradiction",
        "split three ways",
        "split two to one",
    ]
    assert got[0] == {
        "text": "sure contradiction",
        "gold": "neu",
        "teacher_label": "neg",
        "teacher_confidence": "0.9",
        "member_labels": "neg; neg; neg",
    }
    assert got[2]["gold"] == "" and got[2]["member_labels"] == "pos; neg; neu"
    assert "2 whose gold label the teacher contradicts, then 2 on which" in capsys.readouterr().err
    assert not (run / "analysis.html").exists()


def _label_twice(tmp_path, first: list[str]):
    """Label a run with the extra flags ``first``, then again into it without them; returns the run."""
    (tmp_path / "task.yaml").write_text(TASK.replace("  {labels}\n", "  {labels}\n  {examples}\n"))
    words = ["good", "bad"]
    rows = [
        {"review": f"it was {w} {i}", "sentiment": ["pos", "neg"][j]}
        for i in range(2)
        for j, w in enumerate(words)
    ]
    write_jsonl(tmp_path / "d.jsonl", rows)
    run = tmp_path / "run"
    base = ["label", str(tmp_path / "task.yaml"), "--data", str(tmp_path / "d.jsonl"), "--out", str(run)]
    base += ["--teacher", TINY_LM, "--device", "cpu"]
    main([*base, *first])
    main(base)
    return run


def test_label_without_few_shot_removes_the_old_examples(tmp_path):
    from tinydistill.cli import _run_teacher

    run = _label_twice(tmp_path, ["--few-shot", "1"])
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    assert not (run / "examples.jsonl").exists()  # eval and the cascade would show them to the teacher
    assert _run_teacher(run, "cpu").fingerprint() == meta["fingerprint"]


def test_label_without_test_removes_the_old_test_set(tmp_path):
    run = _label_twice(tmp_path, ["--test", str(tmp_path / "d.jsonl")])
    assert not (run / "test.jsonl").exists()  # eval would report on the earlier teacher's test labels


def test_refresh_never_trains_on_test_texts(tmp_path, monkeypatch):
    from tinydistill import cli

    run = tmp_path / "run"
    run.mkdir()
    (run / "task.yaml").write_text(TASK)
    (run / "meta.json").write_text(json.dumps({"teacher": TINY_LM}))
    write_jsonl(run / "train.jsonl", [{"text": "a training text", "gold": None, "scores": [0.0, -1.0]}])
    write_jsonl(run / "test.jsonl", [{"text": "a test text", "gold": 0, "scores": [0.0, -1.0]}])
    logged = [{"text": t, "scores": [0.0, -1.0]} for t in ("a test text", "a new text")]
    write_jsonl(tmp_path / "log.jsonl", logged)
    for step in ("cmd_train", "cmd_export", "cmd_eval"):
        monkeypatch.setattr(cli, step, lambda a: None)
    main(["refresh", str(run), "--log", str(tmp_path / "log.jsonl"), "--out", str(tmp_path / "new")])
    lines = (tmp_path / "new" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["text"] for line in lines] == ["a training text", "a new text"]


def _tune_data(tmp_path) -> str:
    write_jsonl(
        tmp_path / "d.jsonl", [{"review": f"x {i}", "sentiment": ["pos", "neg"][i % 2]} for i in range(4)]
    )
    return str(tmp_path / "d.jsonl")


def test_tune_refuses_settings_it_cannot_score(tmp_path):
    (tmp_path / "task.yaml").write_text(TASK)  # no {examples} line
    args = ["tune", str(tmp_path / "task.yaml"), "--data", _tune_data(tmp_path), "--teacher", TINY_LM]
    with pytest.raises(SystemExit, match="examples"):
        main([*args, "--few-shot", "2", "--device", "cpu", "--out", str(tmp_path / "tune")])


def test_tune_refuses_prompt_files_with_the_same_name(tmp_path):
    for d in ("a", "b"):  # rows name a prompt by its file name, so the best could not be told apart
        (tmp_path / d).mkdir()
        (tmp_path / d / "task.yaml").write_text(TASK)
    args = ["tune", str(tmp_path / "a" / "task.yaml"), "--prompt", str(tmp_path / "b" / "task.yaml")]
    with pytest.raises(SystemExit, match="file names"):
        main([*args, "--data", _tune_data(tmp_path), "--teacher", TINY_LM, "--out", str(tmp_path / "t")])


def test_refresh_trains_with_the_runs_settings(tmp_path, monkeypatch):
    from tinydistill import cli

    run = tmp_path / "run"
    run.mkdir()
    (run / "task.yaml").write_text(TASK)
    meta = {"teacher": TINY_LM, "student": TINY_ENCODER, "epochs": 7, "seeds": 2, "init_head": "names"}
    (run / "meta.json").write_text(json.dumps(meta))
    write_jsonl(run / "train.jsonl", [{"text": "a training text", "gold": None, "scores": [0.0, -1.0]}])
    write_jsonl(tmp_path / "log.jsonl", [{"text": "a new text", "scores": [0.0, -1.0]}])
    trained = []
    monkeypatch.setattr(cli, "cmd_train", trained.append)
    for step in ("cmd_export", "cmd_eval"):
        monkeypatch.setattr(cli, step, lambda a: None)
    args = ["refresh", str(run), "--log", str(tmp_path / "log.jsonl")]
    main([*args, "--out", str(tmp_path / "new")])
    a = trained[-1]
    assert (a.student, a.epochs, a.seeds, a.init_head) == (TINY_ENCODER, 7, 2, "names")
    main([*args, "--out", str(tmp_path / "new2"), "--epochs", "3"])  # a flag given on the command line wins
    assert (trained[-1].epochs, trained[-1].seeds) == (3, 2)
