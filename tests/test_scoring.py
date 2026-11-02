import copy
import math

import pytest
import torch
from conftest import TINY_LM
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinydistill.teacher import LabelScorer, surface_variants


@pytest.fixture(scope="module")
def lm():
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_pretrained(TINY_LM).eval(), AutoTokenizer.from_pretrained(TINY_LM)


def logsumexp_by_label(scorer: LabelScorer, form_logps: list[float]) -> list[float]:
    out = []
    for li in range(len(scorer.task.labels)):
        xs = [x for x, owner in zip(form_logps, scorer.form_label, strict=True) if owner == li]
        m = max(xs)
        out.append(m + math.log(sum(math.exp(x - m) for x in xs)))
    return out


@torch.no_grad()
def reference(scorer: LabelScorer, text: str) -> list[float]:
    """Hand computation: one full forward pass per continuation over prompt + continuation, no cache, then
    the sum of log p(token | everything before it) over the continuation's tokens."""
    prompt = scorer.prefix_ids + scorer.text_ids(text)
    out = []
    for form in scorer.forms:
        ids = torch.tensor([prompt + form])
        logp = torch.log_softmax(scorer.model(ids).logits[0].float(), -1)
        out.append(sum(logp[len(prompt) + j - 1, t].item() for j, t in enumerate(form)))
    return out


@torch.no_grad()
def previous_implementation(scorer: LabelScorer, text: str, batch: int = 2) -> list[float]:
    """The scorer before tree attention (v0.1.0), applied to every continuation: extend a copy of the prompt
    cache with the text, then repeat it per continuation and score right-padded batches."""
    model, dev = scorer.model, scorer.device
    cache = model(torch.tensor([scorer.prefix_ids]), use_cache=True).past_key_values
    out = model(torch.tensor([scorer.text_ids(text)]), past_key_values=cache, use_cache=True)
    first_lp = torch.log_softmax(out.logits[0, -1].float(), -1)
    scores: list[float] = []
    for start in range(0, len(scorer.forms), batch):
        chunk = scorer.forms[start : start + batch]
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


@pytest.mark.parametrize("chat", [True, False])
@pytest.mark.parametrize("forms", [True, False])
def test_multi_token_scores_match_hand_computation(lm, task, chat, forms):
    model, tok = lm
    scorer = LabelScorer(model, tok, task, use_chat_template=chat, surface_forms=forms)
    assert scorer.chat is chat
    assert max(len(ids) for ids in scorer.forms) >= 4  # genuinely multi-token labels
    assert len({len(ids) for ids in scorer.forms}) > 1  # of different lengths
    if not forms:
        assert scorer.form_label == list(range(len(task.labels)))  # one continuation per label
    for text in ["great product, would buy again", "meh", ""]:
        want = reference(scorer, text)
        assert scorer.form_scores(text).tolist() == pytest.approx(want, abs=1e-4)
        assert scorer.score(text) == pytest.approx(logsumexp_by_label(scorer, want), abs=1e-4)


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
    assert len(scorer._node_tokens) < sum(len(ids) - 1 for ids in scorer.forms)  # nodes are shared
    for text in ["where is my card", "x" * 50]:
        assert scorer.form_scores(text).tolist() == pytest.approx(
            previous_implementation(scorer, text), abs=1e-4
        )
        # repeated calls reuse the cropped prompt cache and give the same scores
        assert scorer.score(text) == pytest.approx(scorer.score(text), abs=1e-6)


def test_surface_variants_and_deduplication(lm, task):
    assert surface_variants("card arrival") == [
        "card arrival",
        " card arrival",
        "Card arrival",
        " Card arrival",
        "CARD ARRIVAL",
        " CARD ARRIVAL",
    ]
    assert surface_variants("Visa or Mastercard")[:2] == ["Visa or Mastercard", " Visa or Mastercard"]
    assert surface_variants("card arrival", ("lowercase", "capitalised")) == [
        "card arrival",
        " card arrival",
        "Card arrival",
        " Card arrival",
    ]
    assert surface_variants("Card", ()) == ["Card", " Card"]
    model, tok = lm
    scorer = LabelScorer(model, tok, task)
    assert len(set(map(tuple, scorer.forms))) == len(scorer.forms)  # identical tokenisations kept once
    assert all(ids[-1] == tok.eos_token_id for ids in scorer.forms)


class Biased(torch.nn.Module):
    """Wraps a causal LM and adds a fixed bias to its logits."""

    def __init__(self, model, bias):
        super().__init__()
        self.model, self.bias, self.config = model, bias, model.config

    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        out.logits = out.logits + self.bias
        return out


def test_label_gets_the_mass_of_its_capitalised_form(lm, task):
    model, tok = lm
    two = type(task)(
        **{**task.__dict__, "labels": ["pos", "neg"], "verbalisations": ["positive", "negative"]}
    )
    capital = tok("Positive", add_special_tokens=False)["input_ids"]
    lower = tok("positive", add_special_tokens=False)["input_ids"]
    assert capital[0] != lower[0]  # the forms start differently
    bias = torch.zeros(model.config.vocab_size)
    bias[capital + [tok.eos_token_id]] = 30.0  # the toy model wants to write "Positive"
    toy = Biased(model, bias).eval()

    exact = torch.tensor(LabelScorer(toy, tok, two, surface_forms=False).score("a review")).softmax(-1)
    covered = torch.tensor(LabelScorer(toy, tok, two).score("a review")).softmax(-1)
    assert covered[0] > 0.99  # with surface forms, "positive" gets the mass the model puts on "Positive"
    assert exact[0] < 0.9  # scoring only the name as written misses most of it


def test_labels_sharing_a_surface_form_are_rejected(lm, task):
    model, tok = lm
    clash = type(task)(**{**task.__dict__, "labels": ["a", "b"], "verbalisations": ["yes", "Yes"]})
    with pytest.raises(ValueError, match="share the surface form"):
        LabelScorer(model, tok, clash)


def test_fingerprint_changes_with_the_prompt(lm, task):
    model, tok = lm
    other = type(task)(**{**task.__dict__, "prompt": task.prompt.replace("Review", "Text")})
    assert LabelScorer(model, tok, task).fingerprint() != LabelScorer(model, tok, other).fingerprint()


def test_4bit_needs_cuda(task):
    from tinydistill.teacher import load_teacher

    with pytest.raises(SystemExit, match="CUDA"):
        load_teacher(TINY_LM, task, device="cpu", load_4bit=True)


def test_task_can_drop_uppercase_forms(lm, task):
    model, tok = lm
    full = LabelScorer(model, tok, task)
    lean = LabelScorer(
        model, tok, type(task)(**{**task.__dict__, "surface_forms": ("lowercase", "capitalised")})
    )
    names_only = LabelScorer(model, tok, type(task)(**{**task.__dict__, "surface_forms": ()}))
    assert len(names_only.forms) < len(lean.forms) < len(full.forms)
    upper = {tuple(tok(v.upper(), add_special_tokens=False)["input_ids"]) for v in task.verbalisations}
    assert not any(tuple(f[:-1]) in upper for f in lean.forms)
    assert lean.fingerprint() != full.fingerprint()


@pytest.mark.parametrize("max_tokens", [1, 60, 10_000])  # one text per pass, a few, all of them
def test_packing_several_texts_matches_one_at_a_time(lm, task, max_tokens):
    model, tok = lm
    texts = ["short", "a somewhat longer text with more tokens in it", "", "mid length text here", "x"]
    one = LabelScorer(model, tok, task)
    packed = LabelScorer(model, tok, task, max_tokens=max_tokens)
    want = torch.stack([one.form_scores(t) for t in texts])
    torch.testing.assert_close(packed.form_scores_many(texts), want, atol=1e-4, rtol=0)
    torch.testing.assert_close(
        torch.tensor(packed.score_many(texts)), torch.tensor([one.score(t) for t in texts]), atol=1e-4, rtol=0
    )


def test_packing_with_empty_texts_and_no_suffix(lm, task):
    # Without a chat template and with {text} at the very end, an empty text has no tokens at all, so the
    # prompt itself predicts the first label token.
    model, tok = lm
    bare = type(task)(**{**task.__dict__, "prompt": "Options:\n{labels}\nReview: {text}"})
    scorer = LabelScorer(model, tok, bare, use_chat_template=False, max_tokens=10_000)
    assert scorer.suffix == "" and scorer.text_ids("") == []
    texts = ["", "good", ""]
    got = scorer.form_scores_many(texts)
    for t, row in zip(texts, got, strict=True):
        assert row.tolist() == pytest.approx(reference(scorer, t), abs=1e-4)


def test_few_shot_examples_go_into_the_cached_prompt(lm, task):
    model, tok = lm
    shots = type(task)(
        **{**task.__dict__, "prompt": "Options:\n{labels}\n{examples}\n\nReview: {text}\nAnswer:"}
    )
    plain = LabelScorer(model, tok, shots)
    few = LabelScorer(model, tok, shots, examples=[("loved every minute", 0), ("dull and slow", 1)])
    assert len(few.prefix_ids) > len(plain.prefix_ids)
    assert "loved every minute" in tok.decode(few.prefix_ids)
    assert few.fingerprint() != plain.fingerprint()
    assert few.form_scores("fine").tolist() == pytest.approx(reference(few, "fine"), abs=1e-4)


def test_a_prompt_longer_than_the_teacher_is_rejected(lm, task):
    model, tok = lm
    shots = type(task)(**{**task.__dict__, "prompt": "{labels}\n{examples}\n{text}"})
    many = [("word " * 50, i % 3) for i in range(model.config.max_position_embeddings // 10)]
    with pytest.raises(ValueError, match="more than the teacher"):
        LabelScorer(model, tok, shots, examples=many)


@pytest.mark.parametrize(
    "arch", ["MistralForCausalLM", "PhiForCausalLM", "Gemma2ForCausalLM", "GPT2LMHeadModel"]
)
def test_other_teacher_architectures(arch, task):
    """Tree and packed scoring are exact for other model families too, so any causal LM can be the teacher."""
    name = f"hf-internal-testing/tiny-random-{arch}"
    model, tok = AutoModelForCausalLM.from_pretrained(name).eval(), AutoTokenizer.from_pretrained(name)
    scorer = LabelScorer(model, tok, task, max_tokens=10_000)
    texts = ["great product, would buy again", "meh"]
    got = scorer.form_scores_many(texts)
    for t, row in zip(texts, got, strict=True):
        assert row.tolist() == pytest.approx(reference(scorer, t), abs=1e-3)


def test_ensemble_averages_probabilities():
    from tinydistill.teacher import combine_scores

    one = [0.0, -1.0, -2.0]
    assert combine_scores([one]) == one  # a single teacher is passed through unchanged
    # teacher 1: [0.5, 0.5]; teacher 2: [0.75, 0.25] -> mean [0.625, 0.375]
    got = combine_scores([[0.0, 0.0], [math.log(3.0), 0.0]])
    assert got == pytest.approx([math.log(0.625), math.log(0.375)], abs=1e-12)


def test_chunked_prompt_prefix_gives_the_same_scores(lm, task):
    model, tok = lm
    whole = LabelScorer(model, tok, task)
    chunked = LabelScorer(model, tok, task)
    chunked._encode_prefix(chunk=3)  # many small chunks
    assert len(chunked.prefix_ids) > 6
    for text in ["fine", "a longer review that says rather more"]:
        assert chunked.form_scores(text).tolist() == pytest.approx(whole.form_scores(text).tolist(), abs=1e-4)


@pytest.fixture
def multi_task(task):
    return type(task)(
        **{
            **task.__dict__,
            "labels": ["price", "delivery", "quality"],
            "verbalisations": ["the price", "delivery", "product quality"],
            "multi_label": True,
            "prompt": "Review: {text}\nDoes the review talk about {label}? Answer yes or no.",
        }
    )


@pytest.mark.parametrize("chat", [True, False])
def test_multi_label_scores_are_yes_no_log_odds(lm, multi_task, chat):
    model, tok = lm
    scorer = LabelScorer(model, tok, multi_task, use_chat_template=chat, max_tokens=10_000)
    assert set(scorer.form_answer) == {0, 1} and set(scorer.form_label) == {0, 1, 2}
    texts = ["cheap but it broke after a week", "fine"]
    for text, row, got in zip(texts, scorer.form_scores_many(texts), scorer.score_many(texts), strict=True):
        want = reference(scorer, text)
        assert row.tolist() == pytest.approx(want, abs=1e-4)
        for li in range(3):  # log P(yes forms) - log P(no forms), from the hand computation
            yes = [
                w
                for w, lab, a in zip(want, scorer.form_label, scorer.form_answer, strict=True)
                if lab == li and a == 0
            ]
            no = [
                w
                for w, lab, a in zip(want, scorer.form_label, scorer.form_answer, strict=True)
                if lab == li and a == 1
            ]
            assert got[li] == pytest.approx(
                math.log(sum(map(math.exp, yes))) - math.log(sum(map(math.exp, no))), abs=1e-4
            )


def test_multi_label_ensemble_averages_per_label_probabilities():
    from tinydistill.teacher import combine_scores

    # label 1: p = 0.5 and 0.9 -> 0.7; label 2: p = 0.2 and 0.2 -> 0.2
    got = combine_scores([[0.0, math.log(0.25)], [math.log(9.0), math.log(0.25)]], multi_label=True)
    assert got == pytest.approx([math.log(0.7 / 0.3), math.log(0.25)], abs=1e-12)
