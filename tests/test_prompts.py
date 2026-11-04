"""Prompt ensembles (label --prompts) and instruction search (tune --propose)."""

import json

import pytest
from conftest import TINY_LM
from test_cli import TASK, write_jsonl

from tinydistill.cli import _run_teacher, main
from tinydistill.task import load_task

VARIANT = TASK.replace("Classify the sentiment of the review as one of:", "Which of these fits the review?")


def _data(tmp_path):
    rows = [
        {"review": f"it was {w}", "sentiment": "pos" if i % 2 == 0 else "neg"}
        for i, w in enumerate(["good", "bad", "great", "awful", "fine", "poor"])
    ]
    write_jsonl(tmp_path / "d.jsonl", rows)
    return ["--data", str(tmp_path / "d.jsonl"), "--teacher", TINY_LM, "--device", "cpu"]


def test_prompt_variants_must_match_the_task(tmp_path):
    (tmp_path / "task.yaml").write_text(TASK)
    (tmp_path / "swapped.yaml").write_text(TASK.replace("  pos: positive review\n", "") + "  pos: positive\n")
    (tmp_path / "multi.yaml").write_text(
        TASK.replace("Review: {text}", "Review: {text}\n  Is it {label}?") + "multi_label: true\n"
    )
    (tmp_path / "ok.yaml").write_text(VARIANT)
    label = ["label", str(tmp_path / "task.yaml"), *_data(tmp_path), "--out", str(tmp_path / "run")]
    with pytest.raises(SystemExit, match="label ids in the same order: pos, neg"):
        main([*label, "--prompts", str(tmp_path / "swapped.yaml")])
    with pytest.raises(SystemExit, match="multi_label setting"):
        main([*label, "--prompts", str(tmp_path / "multi.yaml")])
    with pytest.raises(SystemExit, match="needs a local teacher"):
        main([*label, "--prompts", str(tmp_path / "ok.yaml"), "--vllm-url", "http://localhost:1/v1"])


def test_prompt_ensemble_label_eval_and_cascade(tmp_path, capsys):
    from tinydistill.teacher import Ensemble, combine_scores, load_teacher

    (tmp_path / "task.yaml").write_text(TASK)
    (tmp_path / "b.yaml").write_text(VARIANT)
    (tmp_path / "c.yaml").write_text(TASK.replace("Review: {text}", "The review: {text}"))
    data = _data(tmp_path)
    run = tmp_path / "run"
    cache = ["--cache-dir", str(tmp_path / "cache"), "--test", str(tmp_path / "d.jsonl")]
    first = ["--prompts", str(tmp_path / "b.yaml"), "--out", str(tmp_path / "first")]
    main(["label", str(tmp_path / "task.yaml"), *data, *cache, *first])
    capsys.readouterr()
    # one more variant: only that one is scored, the others come from their caches
    prompts = f"{tmp_path / 'b.yaml'},{tmp_path / 'c.yaml'}"
    main(["label", str(tmp_path / "task.yaml"), *data, *cache, "--prompts", prompts, "--out", str(run)])
    err = capsys.readouterr().err
    assert "train: 0 texts scored now by tiny-random-LlamaForCausalLM, prompt 1" in err
    assert "train: 6 texts scored now by tiny-random-LlamaForCausalLM, prompt 2" in err

    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    assert meta["prompts"] == ["prompt-1.yaml", "prompt-2.yaml"] and len(meta["fingerprints"]) == 3
    assert len(set(meta["fingerprints"])) == 3  # the prompt is part of each member's fingerprint
    assert load_task(run / "prompt-1.yaml").prompt == load_task(tmp_path / "b.yaml").prompt

    rows = [json.loads(line) for line in (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    texts = [r["text"] for r in rows]
    variant = load_teacher(TINY_LM, load_task(tmp_path / "b.yaml"), device="cpu").score_many(texts)
    for r, s in zip(rows, variant, strict=True):
        assert len(r["teacher_scores"]) == 3
        assert r["teacher_scores"][1] == pytest.approx(s, abs=1e-5)  # the member is that prompt alone
        assert r["scores"] == pytest.approx(combine_scores(r["teacher_scores"]), abs=1e-9)

    teacher = _run_teacher(run, "cpu")  # the cascade's teacher, rebuilt from the run directory
    assert isinstance(teacher, Ensemble) and len(teacher.members) == 3
    for r, s in zip(rows, teacher.score_many(texts), strict=True):
        assert s == pytest.approx(r["scores"], abs=1e-5)

    main(["eval", str(run), "--teacher-reps", "1", "--reps", "1", "--device", "cpu"])
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert results["models"][0]["name"] == "teacher, zero-shot, 3 prompts"
    assert results["models"][0]["latency_ms_b1"] > 0


def test_tune_proposes_instructions(tmp_path, monkeypatch, capsys):
    import tinydistill.tune

    described = TASK.replace("pos: positive review", "pos: {name: positive, description: liked it}")
    (tmp_path / "task.yaml").write_text(described)
    good = "Sort the review into one of:\n{labels}\n\nReview: {text}"
    replies = iter(
        [
            f"```text\n{good}\n```",  # kept, without the code fence
            "Pick one of {labels}.",  # no {text}
            "{labels}\nReview: {text} {text}",  # {text} twice
            "{labels}\n{examples}\nReview: {text}",  # a placeholder the task does not have
            good,  # a repeat
            "Instruction: Which sentiment does the review have?\n{labels}\nReview: {text}",  # kept
        ]
    )
    monkeypatch.setattr(tinydistill.tune, "generator", lambda model, tok, **kw: lambda msg: next(replies))
    out = tmp_path / "tune"
    main(
        ["tune", str(tmp_path / "task.yaml"), *_data(tmp_path), "--few-shot", "0", "--dev-size", "6"]
        + ["--propose", "6", "--out", str(out)]
    )
    assert "kept 2, dropped 4" in capsys.readouterr().err

    results = json.loads((tmp_path / "tune.json").read_text(encoding="utf-8"))
    names = ["task.yaml", "tune-1.yaml", "tune-2.yaml", "tune-names-only.yaml"]
    assert sorted(r["prompt"] for r in results) == names
    task = load_task(tmp_path / "task.yaml")
    assert load_task(tmp_path / "tune-1.yaml").prompt == good
    assert load_task(tmp_path / "tune-2.yaml").prompt.startswith("Which sentiment")
    names_only = load_task(tmp_path / "tune-names-only.yaml")
    assert names_only.prompt == task.prompt and names_only.descriptions == [] and task.descriptions
    assert names_only.labels == task.labels
    assert load_task(tmp_path / "tune-best.yaml").prompt == results[0]["instruction"]
    md = (tmp_path / "tune.md").read_text(encoding="utf-8")
    assert "### Instructions" in md and good in md and "`tune-names-only.yaml`" in md
