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


def test_example_sets_label_eval_and_cascade(tmp_path, capsys):
    from conftest import TINY_ENCODER

    from tinydistill.teacher import Ensemble, combine_scores

    (tmp_path / "task.yaml").write_text(TASK.replace("  {labels}\n", "  {labels}\n  {examples}\n"))
    data = _data(tmp_path)
    run = tmp_path / "run"
    label = ["label", str(tmp_path / "task.yaml"), *data, "--test", str(tmp_path / "d.jsonl")]
    label += ["--few-shot-similar", "2", "--few-shot-model", TINY_ENCODER, "--cache-dir", str(tmp_path / "c")]
    main([*label, "--out", str(tmp_path / "one")])  # one set: --few-shot-similar as before
    main([*label, "--few-shot-sets", "2", "--out", str(run)])
    err = capsys.readouterr().err
    assert "train: 6 texts scored now by tiny-random-LlamaForCausalLM, set 1" in err
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    one = json.loads((tmp_path / "one" / "meta.json").read_text(encoding="utf-8"))
    assert meta["few_shot_sets"] == 2 and one["few_shot_sets"] == 1 and len(one["fingerprints"]) == 1
    assert len(set(meta["fingerprints"])) == 2 and one["fingerprint"] not in meta["fingerprints"]

    rows = [json.loads(line) for line in (run / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rows:
        assert len(r["teacher_scores"]) == 2
        assert r["scores"] == pytest.approx(combine_scores(r["teacher_scores"]), abs=1e-9)
    teacher = _run_teacher(run, "cpu")  # the cascade's (and refresh's, improve's) teacher
    assert (
        isinstance(teacher, Ensemble) and [m.fingerprint() for m in teacher.members] == meta["fingerprints"]
    )
    for r, s in zip(rows, teacher.score_many([r["text"] for r in rows]), strict=True):
        assert s == pytest.approx(r["scores"], abs=1e-5)

    main([*label, "--few-shot-sets", "2", "--out", str(run)])  # both sets from their caches
    assert "every text is in the cache, so the teacher is not loaded" in capsys.readouterr().err
    assert json.loads((run / "meta.json").read_text(encoding="utf-8"))["fingerprints"] == meta["fingerprints"]

    main(["eval", str(run), "--teacher-reps", "1", "--reps", "1", "--device", "cpu"])
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    assert results["models"][0]["name"] == "teacher, 2 similar examples, 2 example sets"
    assert results["training"]["few_shot_sets"] == 2
    assert "scores each text 2 times, each prompt showing it another 2 of its 4 most similar of 6" in (
        run / "results.md"
    ).read_text(encoding="utf-8")


def test_example_sets_need_similar_examples_and_enough_of_them(tmp_path):
    (tmp_path / "task.yaml").write_text(TASK.replace("  {labels}\n", "  {labels}\n  {examples}\n"))
    label = ["label", str(tmp_path / "task.yaml"), *_data(tmp_path), "--out", str(tmp_path / "run")]
    with pytest.raises(SystemExit, match="needs --few-shot-similar"):
        main([*label, "--few-shot-sets", "2"])
    with pytest.raises(SystemExit, match="needs --few-shot-similar"):
        main([*label, "--few-shot", "1", "--few-shot-sets", "2"])
    with pytest.raises(SystemExit, match="positive"):
        main([*label, "--few-shot-similar", "2", "--few-shot-sets", "0"])
    with pytest.raises(SystemExit, match="needs more than 6 gold examples .* there are 6"):
        main([*label, "--few-shot-similar", "2", "--few-shot-sets", "3"])


def test_tune_proposes_instructions(tmp_path, monkeypatch, capsys):
    import tinydistill.tune

    described = TASK.replace("pos: positive review", "pos: {name: positive, description: liked it}")
    (tmp_path / "task.yaml").write_text(described)
    rest = "{labels}\n\nReview: {text}\n"  # after the lead "Classify the sentiment ... as one of:"
    replies = iter(
        [
            "```text\nSort the review into one of:\n```",  # kept, without the code fence
            "",  # empty
            "x" * 700,  # too long
            "Pick one of {labels}.",  # a placeholder
            "Choose one of these:\n[the list of categories]",  # a placeholder, as the request showed it
            "Is it a positive review or a negative review?",  # kept: with two labels a lead may name both
            "Sort the review into one of:",  # a repeat
            "Instruction: Which sentiment does the review have?",  # kept
        ]
    )
    asked: list[str] = []
    monkeypatch.setattr(
        tinydistill.tune,
        "generator",
        lambda model, tok, **kw: lambda msg: asked.append(msg) or next(replies),
    )
    out = tmp_path / "tune"
    main(
        ["tune", str(tmp_path / "task.yaml"), *_data(tmp_path), "--few-shot", "0", "--dev-size", "6"]
        + ["--propose", "8", "--out", str(out)]
    )
    err = capsys.readouterr().err
    assert "kept 3, dropped 5" in err
    for why in ["empty reply", "reply longer than 600 characters", "an instruction already tried"]:
        assert err.count(f"dropped: {why}\n") == 1
    assert err.count("dropped: reply contains a placeholder\n") == 2
    assert len(set(asked)) == 1 and "{" not in asked[0]  # the request has no placeholders to lose
    assert "[the list of categories]\n\nReview: [the text to classify]" in asked[0]
    assert asked[0].endswith("Instruction to rewrite:\nClassify the sentiment of the review as one of:")

    results = json.loads((tmp_path / "tune.json").read_text(encoding="utf-8"))
    names = ["task.yaml", "tune-1.yaml", "tune-2.yaml", "tune-3.yaml", "tune-names-only.yaml"]
    assert sorted(r["prompt"] for r in results) == names
    task = load_task(tmp_path / "task.yaml")
    good = "Sort the review into one of:\n" + rest  # the reply, then the rest as written
    assert load_task(tmp_path / "tune-1.yaml").prompt == good
    assert (
        load_task(tmp_path / "tune-2.yaml").prompt == "Is it a positive review or a negative review?\n" + rest
    )
    assert load_task(tmp_path / "tune-3.yaml").prompt == "Which sentiment does the review have?\n" + rest
    names_only = load_task(tmp_path / "tune-names-only.yaml")
    assert names_only.prompt == task.prompt and names_only.descriptions == [] and task.descriptions
    assert names_only.labels == task.labels
    assert load_task(tmp_path / "tune-best.yaml").prompt == results[0]["instruction"]
    assert all("round" not in r for r in results)
    md = (tmp_path / "tune.md").read_text(encoding="utf-8")
    assert "### Instructions" in md and good.strip() in md and "`tune-names-only.yaml`" in md
    assert "change only the instruction before the first placeholder" in md


def test_propose_splits_at_the_first_placeholder_or_rewrites_the_whole_prompt(tmp_path):
    from pathlib import Path

    from tinydistill.tune import _mistake, propose, split

    ag = load_task(Path(__file__).parents[1] / "examples/ag-news/task.yaml")
    lead, rest = split(ag.prompt)  # the one-word heading before the list stays with it
    assert lead == "Here is a news headline with the start of the article. What is it about?\n"
    assert rest.startswith("Options:\n{labels}\n\nNews: {text}")

    # no lead: the whole prompt is rewritten, and must keep its placeholders
    no_lead = TASK.replace("Classify the sentiment of the review as one of:\n  ", "")
    (tmp_path / "task.yaml").write_text(no_lead)
    task = load_task(tmp_path / "task.yaml")
    assert split(task.prompt)[0] == ""
    asked: list[str] = []
    replies = iter(["Pick one:\n{labels}\nReview: {text}", "Pick one of {labels}."])
    logged: list[str] = []
    out = propose(task, lambda msg: asked.append(msg) or next(replies), 2, logged.append)
    assert [t.prompt for _, t in out] == ["Pick one:\n{labels}\nReview: {text}"]
    assert "Keep the placeholders {labels}, {text} exactly as written" in asked[0]
    assert logged == [
        "the prompt starts with a placeholder: the teacher rewrites the whole prompt",
        "dropped: reply does not have the placeholders {labels}, {text}, each once",
        "proposed 2 rewrites of the instruction: kept 1, dropped 1",
    ]

    # a reflection request: replies that quote the shown mistakes are dropped
    (tmp_path / "lead.yaml").write_text(TASK)
    task = load_task(tmp_path / "lead.yaml")
    mistakes = _mistake(task, "it was good", 0, 1)
    replies = iter(["Text: it was fine\nCorrect answer: positive review", "Say if it was good.", "Judge it."])
    logged.clear()
    out = propose(task, lambda msg: next(replies), 3, logged.append, mistakes)
    assert [t.prompt for _, t in out] == ["Judge it.\n{labels}\n\nReview: {text}\n"]
    assert logged.count("dropped: reply quotes the mistakes") == 2


def test_tune_rounds_reflect_on_mistakes(tmp_path, monkeypatch):
    import tinydistill.tune
    from tinydistill.teacher import LabelScorer

    (tmp_path / "task.yaml").write_text(TASK)
    data = _data(tmp_path)
    truth = {f"it was {w}": i % 2 for i, w in enumerate(["good", "bad", "great", "awful", "fine", "poor"])}
    asked: list[str] = []

    def score_many(
        self, texts
    ):  # right or wrong by the instruction; OVERFIT is right on the shown texts only
        right = {
            "BETTER": lambda t: t not in asked[2],
            "OVERFIT": lambda t: t in asked[2],
        }
        rule = next((f for w, f in right.items() if w in self.task.prompt), lambda t: False)
        return [[1.0, 0.0] if (truth[t] == 0) == rule(t) else [0.0, 1.0] for t in texts]

    replies = iter(
        [
            "Plain:",  # round 1: wrong everywhere, like the task
            "Pick one of {labels}.",  # dropped
            "BETTER:",  # round 2: right on the scored half
            "OVERFIT:",  # right on the shown half only
            "Again:",  # round 3: no improvement, so no round 4
            "Plain:",  # a repeat of round 1
        ]
    )
    monkeypatch.setattr(LabelScorer, "score_many", score_many)
    monkeypatch.setattr(
        tinydistill.tune,
        "generator",
        lambda model, tok, **kw: lambda msg: asked.append(msg) or next(replies),
    )
    out = tmp_path / "tune"
    main(
        ["tune", str(tmp_path / "task.yaml"), *data, "--few-shot", "0", "--dev-size", "6"]
        + ["--propose", "2", "--rounds", "5", "--out", str(out)]
    )
    assert len(asked) == 6  # three rounds of two requests: round 3 brought no improvement
    assert "Mistakes made with it" not in asked[0]
    shown = [t for t in truth if t in asked[2]]
    assert len(shown) == 3 and asked[2] == asked[3]  # every text of the shown half was wrong
    for t in shown:
        right, wrong = ("positive review", "negative review")[:: 1 - 2 * truth[t]]
        assert f"Text: {t}\nCorrect answer: {right}\nThe model's answer: {wrong}" in asked[2]
    assert "BETTER" in asked[4] and "Plain" not in asked[4]  # round 3 rewrites round 2's best

    results = json.loads((tmp_path / "tune.json").read_text(encoding="utf-8"))
    score = {r["prompt"]: (r["round"], r["accuracy"]) for r in results}
    assert score == {
        "task.yaml": (1, 0.0),
        "tune-1.yaml": (1, 0.0),
        "tune-r2-1.yaml": (2, 1.0),
        "tune-r2-2.yaml": (2, 0.0),  # right on every shown text, but those are not scored
        "tune-r3-1.yaml": (3, 0.0),
    }
    assert load_task(tmp_path / "tune-best.yaml").prompt.startswith("BETTER")
    md = (tmp_path / "tune.md").read_text(encoding="utf-8")
    assert "Scored on 3 gold-labelled" in md and "`tune-r3-1.yaml` (round 3): dev accuracy 0.000" in md

    with pytest.raises(SystemExit, match="--rounds needs --propose"):
        main(["tune", str(tmp_path / "task.yaml"), *data, "--rounds", "2", "--out", str(out)])


def test_a_copied_option_list_is_dropped_from_three_labels_on():
    from tinydistill.task import Task
    from tinydistill.tune import _lead_problem

    three = Task(
        name="t",
        labels=["a", "b", "c"],
        verbalisations=["sports", "politics", "money"],
        prompt="{labels} {text}",
    )
    assert _lead_problem(three, "Is it about sports, politics or money?", 600, "") == (
        "reply contains the whole option list"
    )
    assert _lead_problem(three, "Which topic is the article about?", 600, "") == ""
