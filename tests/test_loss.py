import math

import pytest
import torch

from tinydistill.evaluate import accuracy, macro_f1
from tinydistill.student import distillation_loss


def test_kl_matches_hand_computation():
    # Row 1: KL([.9, .1] || [.5, .5]) = .9 ln(.9/.5) + .1 ln(.1/.5) = 0.368064
    # Row 2: identical distributions, KL = 0. Batch mean = 0.184032.
    logits = torch.tensor([[0.0, 0.0], [math.log(3.0), 0.0]])
    teacher = torch.tensor([[0.9, 0.1], [0.75, 0.25]])
    assert distillation_loss(logits, teacher).item() == pytest.approx(0.184032, abs=1e-6)


def test_cross_entropy_term_only_on_gold_examples():
    logits = torch.tensor([[0.0, 0.0], [math.log(3.0), 0.0]])
    teacher = torch.tensor([[0.9, 0.1], [0.75, 0.25]])
    gold = torch.tensor([0, -1])  # second example has no gold label
    # 0.184032 + 0.5 * (-ln 0.5)
    got = distillation_loss(logits, teacher, gold, ce_weight=0.5).item()
    assert got == pytest.approx(0.184032 + 0.5 * math.log(2), abs=1e-6)
    assert distillation_loss(logits, teacher, gold, ce_weight=0.0).item() == pytest.approx(0.184032, abs=1e-6)


def test_zero_teacher_probabilities_are_safe():
    loss = distillation_loss(torch.tensor([[2.0, -1.0, 0.5]]), torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(
        -torch.log_softmax(torch.tensor([2.0, -1.0, 0.5]), -1)[0].item(), abs=1e-6
    )


def test_metrics_by_hand():
    gold = [0, 0, 1, 1, 2]
    pred = [0, 1, 1, 1, 0]
    assert accuracy(pred, gold) == pytest.approx(0.6)
    # F1: class 0: tp1 fp1 fn1 -> 0.5; class 1: tp2 fp1 fn0 -> 0.8; class 2: tp0 -> 0
    assert macro_f1(pred, gold) == pytest.approx((0.5 + 0.8 + 0.0) / 3)


def test_few_gold_keeps_k_per_class_deterministically():
    from tinydistill.student import few_gold

    gold = [0, 1, 2, 0, 1, 0, None, 2, 0, 1, 0]
    kept = few_gold(gold, 2, seed=3)
    assert kept == few_gold(gold, 2, seed=3)
    assert all(k is None or k == g for k, g in zip(kept, gold, strict=True))  # only drops, never changes
    counts = {c: sum(k == c for k in kept) for c in (0, 1, 2)}
    assert counts == {0: 2, 1: 2, 2: 2}
    assert few_gold(gold, 10) == gold  # k larger than every class keeps everything


def test_few_shot_examples_are_the_few_gold_ones_shuffled():
    from tinydistill.student import few_gold, few_shot_examples

    texts = [f"t{i}" for i in range(12)]
    gold = [0, 1, 2] * 4
    ex = few_shot_examples(texts, gold, 2, seed=5)
    kept = few_gold(gold, 2, seed=5)
    assert sorted(ex) == sorted((t, g) for t, g in zip(texts, kept, strict=True) if g is not None)
    assert ex == few_shot_examples(texts, gold, 2, seed=5)
    assert len(ex) == 6


def test_aggregate_over_seeds():
    from tinydistill.evaluate import aggregate

    one = aggregate([{"accuracy": 0.5}])
    assert one == {"accuracy": 0.5}  # a single seed has no spread
    agg = aggregate([{"accuracy": 0.6, "macro_f1": 0.5}, {"accuracy": 0.8, "macro_f1": 0.7}])
    assert agg["accuracy"] == pytest.approx(0.7) and agg["accuracy_std"] == pytest.approx(
        0.141421356, abs=1e-8
    )
    assert agg["macro_f1_seeds"] == [0.5, 0.7]
    assert aggregate([{}]) == {}
