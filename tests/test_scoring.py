import copy

import pytest
import torch
from conftest import TINY_LM
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinydistill.teacher import LabelScorer


@pytest.fixture(scope="module")
def lm():
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_pretrained(TINY_LM).eval(), AutoTokenizer.from_pretrained(TINY_LM)


@torch.no_grad()
def previous_implementation(scorer: LabelScorer, text: str, label_batch: int = 2) -> list[float]:
    """The scorer before tree attention (v0.1.0): extend a copy of the prompt cache with the text, then
    repeat it per label and score right-padded label batches."""
    model, dev = scorer.model, scorer.device
    cache = model(torch.tensor([scorer.prefix_ids]), use_cache=True).past_key_values
    out = model(torch.tensor([scorer.text_ids(text)]), past_key_values=cache, use_cache=True)
    first_lp = torch.log_softmax(out.logits[0, -1].float(), -1)
    scores: list[float] = []
    for start in range(0, len(scorer.label_ids), label_batch):
        chunk = scorer.label_ids[start : start + label_batch]
        m = max(len(c) for c in chunk)
        tokens = torch.zeros((len(chunk), m), dtype=torch.long, device=dev)
        mask = torch.zeros((len(chunk), m), dtype=torch.bool, device=dev)
        for i, c in enumerate(chunk):
            tokens[i, : len(c)] = torch.tensor(c)
            mask[i, : len(c)] = True
        total = first_lp[tokens[:, 0]]
        c2 = copy.deepcopy(cache)
        c2.batch_repeat_interleave(len(chunk))
        logits = model(tokens[:, :-1], past_key_values=c2, use_cache=True).logits
        lp = torch.log_softmax(logits.float(), -1).gather(-1, tokens[:, 1:, None])[..., 0]
        scores.extend((total + (lp * mask[:, 1:]).sum(-1)).tolist())
    return scores


@torch.no_grad()
def reference(scorer: LabelScorer, text: str) -> list[float]:
    """Hand computation: one full forward pass per label over prompt + label tokens, no cache, then
    sum of log p(token | everything before it) over the label tokens."""
    prompt = scorer.prefix_ids + scorer.text_ids(text)
    out = []
    for label in scorer.label_ids:
        ids = torch.tensor([prompt + label])
        logp = torch.log_softmax(scorer.model(ids).logits[0].float(), -1)
        out.append(sum(logp[len(prompt) + j - 1, t].item() for j, t in enumerate(label)))
    return out


@pytest.mark.parametrize("chat", [True, False])
def test_multi_token_scores_match_hand_computation(lm, task, chat):
    model, tok = lm
    scorer = LabelScorer(model, tok, task, use_chat_template=chat)
    assert scorer.chat is chat
    assert max(len(ids) for ids in scorer.label_ids) >= 4  # genuinely multi-token labels
    assert len({len(ids) for ids in scorer.label_ids}) > 1  # of different lengths (exercises padding)
    for text in ["great product, would buy again", "meh", ""]:
        got = scorer.score(text)
        assert got == pytest.approx(reference(scorer, text), abs=1e-4)


def test_continuations_end_with_eos(lm, task):
    model, tok = lm
    scorer = LabelScorer(model, tok, task)
    assert all(ids[-1] == tok.eos_token_id for ids in scorer.label_ids)


def test_fingerprint_changes_with_the_prompt(lm, task):
    model, tok = lm
    other = type(task)(**{**task.__dict__, "prompt": task.prompt.replace("Review", "Text")})
    assert LabelScorer(model, tok, task).fingerprint() != LabelScorer(model, tok, other).fingerprint()


def test_tree_scoring_equals_previous_implementation(lm, task):
    model, tok = lm
    # Labels that share leading tokens exercise the prefix tree.
    shared = type(task)(
        **{
            **task.__dict__,
            "labels": ["a", "b", "c", "d"],
            "verbalisations": ["card arrival", "card linking", "card about to expire", "cash"],
        }
    )
    scorer = LabelScorer(model, tok, shared)
    assert len(scorer._node_tokens) < sum(len(ids) - 1 for ids in scorer.label_ids)  # nodes are shared
    for text in ["where is my card", "x" * 50]:
        assert scorer.score(text) == pytest.approx(previous_implementation(scorer, text), abs=1e-4)
        # repeated calls reuse the cropped prompt cache and give the same scores
        assert scorer.score(text) == pytest.approx(scorer.score(text), abs=1e-6)


def test_4bit_needs_cuda(task):
    from tinydistill.teacher import load_teacher

    with pytest.raises(SystemExit, match="CUDA"):
        load_teacher(TINY_LM, task, device="cpu", load_4bit=True)
