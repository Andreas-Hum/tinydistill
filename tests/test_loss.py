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


def test_cascade_curve_by_hand():
    from tinydistill.evaluate import cascade_curve

    confidence = [0.9, 0.2, 0.6, 0.4]
    student = [0, 1, 1, 0]  # right, wrong, right, wrong
    teacher = [0, 0, 1, 1]  # all right
    gold = [0, 0, 1, 1]
    c = cascade_curve(confidence, student, teacher, gold, shares=(0.0, 0.5, 1.0))
    assert [x["accuracy"] for x in c] == [0.5, 1.0, 1.0]
    assert [x["share_to_teacher"] for x in c] == [0.0, 0.5, 1.0]
    assert c[1]["threshold"] == 0.6  # texts below 0.6 (the 0.2 and 0.4 ones) go to the teacher


def test_ordinal_metrics_by_hand():
    from tinydistill.evaluate import ordinal_metrics

    m = ordinal_metrics([0, 2, 4, 1], [0, 1, 1, 1])  # distances 0, 1, 3, 0
    assert m == {"mae": 1.0, "within_one": 0.75}


def test_multilabel_metrics_by_hand():
    from tinydistill.evaluate import multilabel_metrics

    gold = [{0, 1}, {1}, set()]
    pred = [{0, 1}, {1, 2}, set()]
    m = multilabel_metrics(pred, gold, 3)
    # exact match: texts 1 and 3. F1: label 0 = 1, label 1 = 1, label 2: tp 0 fp 1 -> 0. micro: tp 3 fp 1 fn 0
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["macro_f1"] == pytest.approx(2 / 3)
    assert m["micro_f1"] == pytest.approx(6 / 7)


def test_multilabel_distillation_loss_and_gold_selection():
    from tinydistill.student import distillation_loss_multilabel, few_gold_sets

    z = torch.zeros(1, 2)
    assert distillation_loss_multilabel(z, torch.tensor([[1.0, 0.0]])).item() == pytest.approx(math.log(2))
    gold = torch.tensor([[1.0, 1.0]])
    both = distillation_loss_multilabel(
        z, torch.tensor([[1.0, 0.0]]), gold, torch.tensor([True]), ce_weight=0.5
    )
    assert both.item() == pytest.approx(math.log(2) * 1.5)
    sets = [[0], [0, 1], [1], None, [0], [2]]
    kept = few_gold_sets(sets, 1, seed=0)
    assert all(k is None or k == s for k, s in zip(kept, sets, strict=True))
    assert {c for k in kept if k for c in k} == {0, 1, 2}  # every label is covered
