import json

import pytest
from conftest import TINY_ENCODER, TINY_LM
from test_cli import TASK, write_jsonl

import tinydistill.synth
from tinydistill.cli import main
from tinydistill.improve import hold_out, jobs
from tinydistill.task import load_task


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_hold_out_keeps_a_documents_windows_together():
    rows = [{"doc": i // 3} for i in range(30)] + [{} for _ in range(10)]
    held = hold_out(rows, 0.2, seed=0)
    assert held == hold_out(rows, 0.2, seed=0) and len({rows[i].get("doc", i) for i in held}) == 4
    assert all({j for j in range(30) if j // 3 == i // 3} <= held for i in held if i < 30)


def test_jobs_target_both_directions_of_the_top_confusions(tmp_path):
    (tmp_path / "task.yaml").write_text(TASK)
    task = load_task(tmp_path / "task.yaml")
    js = jobs(task, ["a", "b", "c", "d"], pred=[1, 1, 0, 0], target=[0, 0, 0, 0])
    assert [li for li, _ in js] == [0, 1]
    assert 'easy to mistake for the category "negative review"' in js[0][1] and "- a\n- b" in js[0][1]
    assert 'yet clearly "negative review"' in js[1][1] and "took texts like these" not in js[1][1]
    assert jobs(task, ["a"], [0], [0]) == []


def test_improve_runs_a_round_without_looking_at_the_test_set(tmp_path, monkeypatch, capsys):
    (tmp_path / "task.yaml").write_text(TASK)
    write_jsonl(tmp_path / "train.jsonl", [{"review": f"train text {i}"} for i in range(16)])
    write_jsonl(
        tmp_path / "test.jsonl",
        [{"review": f"test text {i}", "sentiment": ["pos", "neg"][i % 2]} for i in range(6)],
    )
    run = tmp_path / "run"
    common = ["--student", TINY_ENCODER, "--epochs", "1", "--batch-size", "4", "--device", "cpu"]
    main(
        ["run", str(tmp_path / "task.yaml"), "--data", str(tmp_path / "train.jsonl"), "--test"]
        + [
            str(tmp_path / "test.jsonl"),
            "--out",
            str(run),
            "--teacher",
            TINY_LM,
            "--reps",
            "2",
            "--teacher-reps",
            "2",
        ]
        + common
    )
    rows = read(run / "train.jsonl")  # the teacher's labels alternate, so a tiny student gets some wrong
    write_jsonl(
        run / "train.jsonl", [r | {"scores": [[3.0, -3.0], [-3.0, 3.0]][i % 2]} for i, r in enumerate(rows)]
    )
    before = (run / "train.jsonl").read_text(encoding="utf-8")

    asked = []

    def fake_generator(model, tok, temperature=0.9):
        def generate(message):
            asked.append(message)
            return "\n".join(f"new review {len(asked)} {i}" for i in range(3))

        return generate

    monkeypatch.setattr(tinydistill.synth, "generator", fake_generator)
    capsys.readouterr()
    main(["improve", str(run), "--rounds", "1", "--per-round", "4", "--held-out", "0.25", "--device", "cpu"])
    out = tmp_path / "run-improve"
    assert (run / "train.jsonl").read_text(encoding="utf-8") == before  # the run is left as it was
    held = [r["text"] for r in read(out / "held_out.jsonl")]
    assert len(held) == 4 and set(held) <= {r["text"] for r in rows}
    test_texts = {r["text"] for r in read(run / "test.jsonl")}
    assert asked and not any(t in m for m in asked for t in test_texts)  # never selects on the test set
    table = json.loads((out / "improve.json").read_text(encoding="utf-8"))
    assert [r["round"] for r in table] == [0, 1] and table[0]["added"] == 0 and table[1]["added"] > 0
    assert table[0]["test"] is not None and "| round | texts added |" in (out / "improve.md").read_text()
    for k in (0, 1):
        d = out / f"round-{k}"
        assert (d / "results.json").exists() and (d / "onnx" / "model.onnx").exists()
        trained = [r["text"] for r in read(d / "train.jsonl")]
        assert not set(trained) & set(held) and len(trained) == 12 + table[k]["added"]
    assert read(out / "round-1" / "train.jsonl")[-1]["text"].startswith("new review")
    assert json.loads((out / "round-1" / "meta.json").read_text())["improve_added"] == table[1]["added"]


def test_improve_refuses_multi_label_runs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "task.yaml").write_text(
        "prompt: 'Text: {text} Is it {label}?'\nmulti_label: true\nlabels: [a, b]\n"
    )
    with pytest.raises(SystemExit, match="single-label"):
        main(["improve", str(run)])
