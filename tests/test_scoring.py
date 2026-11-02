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
@pytest.mark.parametrize("label_batch", [32, 2])
def test_multi_token_scores_match_hand_computation(lm, task, chat, label_batch):
    model, tok = lm
    scorer = LabelScorer(model, tok, task, label_batch=label_batch, use_chat_template=chat)
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
