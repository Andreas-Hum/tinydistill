import json

import pytest
import yaml

from tinydistill.task import label_name, load_task

PROMPT = "Pick one:\n{labels}\n\nText: {text}"


def write(tmp_path, labels, prompt=PROMPT):
    p = tmp_path / "task.yaml"
    p.write_text(yaml.safe_dump({"name": "t", "prompt": prompt, "labels": labels}))
    return p


def test_label_names_are_derived_from_ids():
    assert label_name("card_arrival") == "card arrival"
    assert label_name("top-up--limits") == "top up limits"
    assert label_name("visa_or_mastercard ") == "visa or mastercard"


def test_three_label_forms_and_descriptions_in_the_prompt(tmp_path):
    task = load_task(
        write(
            tmp_path,
            {
                "card_arrival": None,
                "lost_card": "lost or stolen card",
                "pin": {"description": "anything about the card's PIN"},
            },
        )
    )
    assert task.labels == ["card_arrival", "lost_card", "pin"]
    assert task.verbalisations == ["card arrival", "lost or stolen card", "pin"]
    assert task.render("hello") == (
        "Pick one:\n- card arrival\n- lost or stolen card\n"
        "- pin: anything about the card's PIN\n\nText: hello"
    )


def test_list_form_and_round_trip(tmp_path):
    task = load_task(write(tmp_path, ["age_limit", "atm_support"]))
    assert task.verbalisations == ["age limit", "atm support"]
    assert task.descriptions == []
    # the run directory stores the task via to_dict(); it must load back identically
    for t in (task, load_task(write(tmp_path, {"a_b": {"name": "x", "description": "y"}, "c": None}))):
        p = tmp_path / "saved.yaml"
        p.write_text(yaml.safe_dump(t.to_dict()))
        assert load_task(p) == t


@pytest.mark.parametrize(
    "labels, prompt, message",
    [
        (["a", "b"], "No list here: {text}", "{labels}"),
        (["a", "b"], "{labels} {text} {text}", "{text}"),
        ({"a_b": None, "a b": None}, PROMPT, "unique"),
        ({"a": {"nmae": "x"}, "b": None}, PROMPT, "unknown keys"),
        (["only_one"], PROMPT, "two labels"),
    ],
)
def test_invalid_tasks_are_rejected(tmp_path, labels, prompt, message):
    with pytest.raises(ValueError, match=message):
        load_task(write(tmp_path, labels, prompt))


def test_surface_forms_option(tmp_path):
    assert load_task(write(tmp_path, ["a", "b"])).surface_forms == ("lowercase", "capitalised", "uppercase")
    p = tmp_path / "t.yaml"
    p.write_text(
        yaml.safe_dump({"prompt": PROMPT, "labels": ["a", "b"], "surface_forms": ["uppercase", "lowercase"]})
    )
    task = load_task(p)
    assert task.surface_forms == ("lowercase", "uppercase")  # canonical order
    (tmp_path / "saved.yaml").write_text(yaml.safe_dump(task.to_dict()))
    assert load_task(tmp_path / "saved.yaml") == task
    p.write_text(yaml.safe_dump({"prompt": PROMPT, "labels": ["a", "b"], "surface_forms": ["shouty"]}))
    with pytest.raises(ValueError, match="surface_forms"):
        load_task(p)


def test_examples_render_in_place_and_vanish_without_any(tmp_path):
    prompt = "Pick one:\n{labels}\n{examples}\n\nText: {text}"
    task = load_task(write(tmp_path, ["pos", "neg"], prompt))
    assert task.render("hi") == "Pick one:\n- pos\n- neg\n\nText: hi"
    assert task.render("hi", [("great\nfilm", 0), ("bad", 1)]) == (
        'Pick one:\n- pos\n- neg\n\nExamples:\n- "great film" -> pos\n- "bad" -> neg\n\nText: hi'
    )


def test_synthetic_texts_are_parsed_deduplicated_and_unlabelled(tmp_path):
    from tinydistill.synth import parse_lines, request, synthesise

    task = load_task(write(tmp_path, {"pos": None, "neg": {"description": "a complaint"}}))
    assert (
        "neg" in request(task, 1, 5)
        and "a complaint" in request(task, 1, 5)
        and "{text}" not in request(task, 1, 5)
    )
    assert parse_lines('1. "Great film"\n- Loved it!\n• meh it was ok\npos\nab\n', task) == [
        "Great film",
        "Loved it!",
        "meh it was ok",
    ]
    replies = iter(["1. great\n2. Great\n3. superb", "4. lovely", "bad\nawful\nterrible\nworse"])
    n = synthesise(
        task, lambda message: next(replies), per_label=3, out=tmp_path / "s.jsonl", log=lambda m: None
    )
    rows = [json.loads(line) for line in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert n == 6 and [r["text"] for r in rows] == ["great", "superb", "lovely", "bad", "awful", "terrible"]
    assert all("label" not in r for r in rows)  # never mistaken for gold labels


def test_label_list_items_can_be_mappings(tmp_path):
    task = load_task(
        write(tmp_path, ["a_b", {"neutral": "no particular emotion"}, {"c": {"description": "d"}}])
    )
    assert task.labels == ["a_b", "neutral", "c"]
    assert task.verbalisations == ["a b", "no particular emotion", "c"]
    assert task.descriptions == [None, None, "d"]


def test_synth_asks_as_often_as_needed(tmp_path):
    from itertools import count

    from tinydistill.synth import synthesise

    task = load_task(write(tmp_path, ["pos", "neg"]))
    n = count()
    batch = lambda message: "\n".join(f"text {next(n)}" for _ in range(10))  # noqa: E731  10 new per request
    written = synthesise(task, batch, per_label=200, out=tmp_path / "s.jsonl", log=lambda m: None)
    assert written == 400  # the old fixed limit of 10 requests stopped at 100 per label
