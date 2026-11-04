"""A teacher that reasons before it answers (label --reason)."""

import json
from dataclasses import replace

import pytest
import torch
from conftest import TINY_LM
from test_cli import TASK, write_jsonl
from test_prompts import VARIANT
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinydistill.cli import _run_teacher, main
from tinydistill.teacher import (
    REASON_INSTRUCTION,
    REASON_MARKER,
    LabelScorer,
    ReasoningScorer,
    cut_reasoning,
    load_teacher,
    quick_key,
    reasoning_task,
)


@pytest.fixture(scope="module")
def lm():
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_pretrained(TINY_LM).eval(), AutoTokenizer.from_pretrained(TINY_LM)


def test_the_generation_is_cut_at_the_marker_or_gets_one():
    assert cut_reasoning(" It is upbeat.\nAnswer: positive\nmore") == (
        "It is upbeat.",
        " It is upbeat.\nAnswer:",
    )
    assert cut_reasoning("It is upbeat. Answer: Answer:") == ("It is upbeat.", "It is upbeat. Answer:")
    assert cut_reasoning("It is upbeat.  \n") == ("It is upbeat.", "It is upbeat.\nAnswer:")
    assert cut_reasoning("") == ("", REASON_MARKER)
    assert cut_reasoning("Answer:") == ("", REASON_MARKER)


def test_the_prompt_asks_for_reasoning(task):
    assert task.prompt.endswith("Answer:")
    assert reasoning_task(task).prompt == task.prompt.removesuffix("\nAnswer:") + REASON_INSTRUCTION
    with pytest.raises(SystemExit, match="single-label"):
        reasoning_task(replace(task, multi_label=True))


@torch.no_grad()
def full_sequence(scorer: LabelScorer, prompt: list[int]) -> list[float]:
    """Each continuation scored by one plain forward pass over the whole sequence, no cache."""
    out = []
    for form in scorer.forms:
        logp = torch.log_softmax(scorer.model(torch.tensor([prompt + form])).logits[0].float(), -1)
        out.append(sum(logp[len(prompt) + j - 1, t].item() for j, t in enumerate(form)))
    return out


@pytest.mark.parametrize("chat", [True, False])
def test_scores_equal_full_sequence_scoring_after_the_reasoning(lm, task, chat, monkeypatch):
    model, tok = lm
    scorer = ReasoningScorer(model, tok, task, 8, use_chat_template=chat)
    plain = LabelScorer(model, tok, reasoning_task(task), use_chat_template=chat)
    texts = ["great product, would buy again", "meh", ""]
    written = [" It sounds happy.\nAnswer: positive", "Hard to say", ""]
    monkeypatch.setattr(scorer, "_generate", lambda prompts: written[: len(prompts)])
    got = scorer.form_scores_many(texts)
    assert scorer.reasoning == dict(zip(texts, ["It sounds happy.", "Hard to say", ""], strict=True))
    for text, out, row in zip(texts, written, got, strict=True):
        tail = tok(cut_reasoning(out)[1], add_special_tokens=False)["input_ids"]
        want = full_sequence(scorer, plain.prefix_ids + plain.text_ids(text) + tail)
        assert row.tolist() == pytest.approx(want, abs=1e-4)


def test_real_generation_then_scoring(lm, task):
    model, tok = lm
    scorer = ReasoningScorer(model, tok, task, 6)
    texts = ["a long review of a product that was fine", "ok"]  # different lengths: left padding
    got = scorer.form_scores_many(texts)
    assert set(scorer.reasoning) == set(texts) and len(got) == 2
    plain = LabelScorer(model, tok, reasoning_task(task))
    for text, row in zip(texts, got, strict=True):
        assert tok.decode(scorer._after[text]).endswith(REASON_MARKER)
        want = full_sequence(scorer, plain.prefix_ids + plain.text_ids(text) + scorer._after[text])
        assert row.tolist() == pytest.approx(want, abs=1e-4)


def test_fingerprints_and_quick_keys_include_the_setting(lm, task):
    model, tok = lm
    prints = {
        LabelScorer(model, tok, task).fingerprint(),
        ReasoningScorer(model, tok, task, 256).fingerprint(),
        ReasoningScorer(model, tok, task, 64).fingerprint(),
    }
    assert len(prints) == 3
    keys = {quick_key(TINY_LM, task, "cpu", False, reason_tokens=k) for k in (0, 64, 256)}
    assert len(keys) == 3
    assert quick_key(TINY_LM, task, "cpu", False) == quick_key(TINY_LM, task, "cpu", False, reason_tokens=0)
    assert type(load_teacher(TINY_LM, task, device="cpu")) is LabelScorer  # without --reason: unchanged
    with pytest.raises(SystemExit, match="--reason needs a local or vLLM teacher"):
        load_teacher("m", task, api_url="http://localhost:1/v1", reason_tokens=8)


def test_label_reason_meta_cache_and_prompt_members(tmp_path, capsys):
    from tinydistill.teacher import Ensemble

    (tmp_path / "task.yaml").write_text(TASK)
    (tmp_path / "b.yaml").write_text(VARIANT)
    rows = [
        {"review": f"it was {w}", "sentiment": "pos" if i % 2 == 0 else "neg"} for i, w in enumerate("abcd")
    ]
    write_jsonl(tmp_path / "d.jsonl", rows)
    run = tmp_path / "run"
    label = [
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
        "--prompts",
        str(tmp_path / "b.yaml"),
    ]
    main([*label, "--reason", "--reason-tokens", "5"])
    assert "reasons in up to 5 tokens" in capsys.readouterr().err
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    assert meta["reason"] is True and meta["reason_tokens"] == 5 and len(meta["fingerprints"]) == 2
    for fp in meta["fingerprints"]:
        cached = [json.loads(line) for line in (run / f"teacher-{fp}.jsonl").read_text().splitlines()]
        assert len(cached) == 4 and all(set(r) == {"h", "s", "r"} for r in cached)
    why = [json.loads(line) for line in (run / "reasoning.jsonl").read_text().splitlines()]
    assert [r["split"] for r in why] == ["train"] * 4 + ["test"] * 4
    assert all(len(r["reasoning"]) == 2 and all(isinstance(x, str) for x in r["reasoning"]) for r in why)

    teacher = _run_teacher(run, "cpu")  # eval and the cascade rebuild the same teacher
    assert isinstance(teacher, Ensemble)
    assert all(isinstance(m, ReasoningScorer) and m.reason_tokens == 5 for m in teacher.members)
    assert [m.fingerprint() for m in teacher.members] == meta["fingerprints"]
    train = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines()]
    for r, s in zip(train, teacher.score_many([r["text"] for r in train]), strict=True):
        assert s == pytest.approx(r["scores"], abs=1e-5)

    main([*label, "--reason", "--reason-tokens", "5"])  # again: all from the cache, reasoning included
    assert "every text is in the cache" in capsys.readouterr().err
    assert [json.loads(line) for line in (run / "reasoning.jsonl").read_text().splitlines()] == why

    main(label)  # without --reason: other fingerprints, plain cache rows, no reasoning file
    meta2 = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    assert meta2["reason"] is False and not set(meta2["fingerprints"]) & set(meta["fingerprints"])
    plain = (run / f"teacher-{meta2['fingerprints'][0]}.jsonl").read_text().splitlines()
    assert all(set(json.loads(line)) == {"h", "s"} for line in plain)
    assert not (run / "reasoning.jsonl").exists()


def test_analyse_shows_the_reasoning_on_teacher_mistakes(tmp_path, task, monkeypatch):
    from tinydistill import analyse

    p = {"task": task, "texts": ["a", "b"], "gold": [0, 1], "teacher": [0, 0], "student": [0, 1]}
    monkeypatch.setattr(analyse, "predictions", lambda run: p)
    page, _ = analyse.build(tmp_path)
    assert "reasoning on its mistakes" not in page
    rows = [{"split": "test", "text": t, "reasoning": [f"thought <{t}>"]} for t in ("a", "b")]
    write_jsonl(tmp_path / "reasoning.jsonl", rows)
    page, _ = analyse.build(tmp_path)
    assert (
        "reasoning on its mistakes" in page
        and "thought &lt;b&gt;" in page
        and "thought &lt;a&gt;" not in page
    )
