import json
import re

import pytest
import torch
from conftest import TINY_LM

from tinydistill.cli import main
from tinydistill.synth import ATTRIBUTES, NearDuplicates, load_attributes, request, synthesise

TASK = """\
name: toy
prompt: |
  Classify the review as one of:
  {labels}

  Review: {text}
labels: [pos, neg]
"""


def recorder():
    asked = []

    def generate(message):
        asked.append(message)
        return "\n".join(f"text {len(asked)} {i}" for i in range(5))

    return asked, generate


def test_vary_samples_attributes_with_the_seed(tmp_path, task):
    runs = []
    for seed in (1, 1, 2):
        asked, generate = recorder()
        synthesise(
            task, generate, 10, tmp_path / "s.jsonl", batch=5, attributes=ATTRIBUTES, seed=seed, log=print
        )
        runs.append(asked)
    assert runs[0] == runs[1] != runs[2]
    assert all("Give every text these attributes: length: " in m and "tone: " in m for m in runs[0])
    rows = [json.loads(line) for line in (tmp_path / "s.jsonl").read_text(encoding="utf-8").splitlines()]
    assert set(rows[0]["attributes"]) == set(ATTRIBUTES)
    assert any(f"tone: {rows[0]['attributes']['tone']};" in m for m in runs[2])

    asked, generate = recorder()  # without --vary the requests are as before
    synthesise(task, generate, 5, tmp_path / "s.jsonl", batch=5, log=print)
    assert asked == [request(task, i, 5) for i in range(3)]
    assert "quotes.\n\nInstruction:" in asked[0] and "attributes" not in asked[0]


def test_task_file_attributes_replace_the_built_in_ones(tmp_path):
    p = tmp_path / "task.yaml"
    p.write_text(TASK)
    assert load_attributes(p) == ATTRIBUTES
    p.write_text(TASK + "attributes:\n  channel: [email, chat]\n")
    assert load_attributes(p) == {"channel": ["email", "chat"]}
    p.write_text(TASK + "attributes:\n  channel: email\n")
    with pytest.raises(SystemExit, match="attributes"):
        load_attributes(p)


def bag_of_words(texts):
    vocab = ["card", "arrive", "late", "top", "up", "money", "my", "the", "has", "not", "did"]
    v = torch.tensor([[float(w in re.findall(r"\w+", t.lower())) for w in vocab] for t in texts]) + 1e-6
    return torch.nn.functional.normalize(v, dim=-1)


def test_dedup_drops_near_duplicates(tmp_path, task):
    keep = NearDuplicates(bag_of_words, 0.95, existing=["top up my money"])
    replies = iter(["my card did not arrive\nMy card did not arrive!\nthe card is late\nup top money my"])
    n = synthesise(
        task, lambda m: next(replies, ""), 4, tmp_path / "s.jsonl", keep=keep, log=print, max_rounds=1
    )
    texts = [
        json.loads(line)["text"] for line in (tmp_path / "s.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    # "My card did not arrive!" has the words of the first; "up top money my" those of the --data text
    assert n == 2 and texts == ["my card did not arrive", "the card is late"] and keep.dropped == 2


def test_synth_command_varies_and_dedups(tmp_path, capsys):
    (tmp_path / "task.yaml").write_text(TASK)
    (tmp_path / "real.jsonl").write_text(json.dumps({"text": "a real review"}) + "\n")
    out = tmp_path / "synth.jsonl"
    main(
        ["synth", str(tmp_path / "task.yaml"), "--per-label", "2", "--out", str(out), "--teacher", TINY_LM]
        + ["--device", "cpu", "--vary", "--dedup", "0.95", "--data", str(tmp_path / "real.jsonl")]
    )
    assert "--dedup: dropped" in capsys.readouterr().err
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert all(set(r) == {"text", "prompted_label", "attributes"} for r in rows)  # a tiny model's gibberish
    with pytest.raises(SystemExit, match="--dedup"):
        main(
            ["synth", str(tmp_path / "task.yaml"), "--per-label", "1", "--out", str(out), "--data", "x.jsonl"]
        )
