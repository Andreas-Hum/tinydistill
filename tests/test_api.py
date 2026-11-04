"""The hosted API teacher against a stand-in server that answers like an OpenAI-compatible chat completions
endpoint (top log-probabilities of the first answer token). No real network calls."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import TINY_ENCODER

from tinydistill import teacher as teacher_mod
from tinydistill.cli import main
from tinydistill.task import Task
from tinydistill.teacher import ApiScorer, load_teacher

KEY_ENV = "TINYDISTILL_TEST_API_KEY"
SECRET = "sk-test-0123456789abcdef"


def top(*pairs):
    return {
        "choices": [
            {"logprobs": {"content": [{"top_logprobs": [{"token": t, "logprob": lp} for t, lp in pairs]}]}}
        ]
    }


def default_answer(content: str) -> dict:
    """Option A for texts with a positive word, B otherwise."""
    text = content.split("Review: ", 1)[1].split("\n", 1)[0]
    good = any(w in text for w in ("good", "great", "fine"))
    return top(("A", -0.2 if good else -2.0), (" B", -2.0 if good else -0.2), ("The", -5.0))


@pytest.fixture
def api():
    state = {"queue": [], "requests": []}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state["requests"].append((self.path, dict(self.headers), body))
                status, payload = (
                    state["queue"].pop(0)
                    if state["queue"]
                    else (200, default_answer(body["messages"][0]["content"]))
                )
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1/", state
    srv.shutdown()


def test_letters_are_scored_from_the_top_logprobs(api, task):
    url, state = api
    state["queue"].append((200, top((" B", -0.1), ("A", -3.0), ("a", -2.0), ("b", -4.0), ("Hello", -9.0))))
    scorer = ApiScorer(url, "some-model", task, api_key_env=KEY_ENV, workers=1)
    # A: the best of "A" and "a"; B: " B"; C is not among the returned tokens: the smallest returned value
    assert scorer.score("nice") == [-2.0, -0.1, -9.0]
    path, _, body = state["requests"][-1]
    assert path == "/v1/chat/completions"
    assert {k: body[k] for k in ("model", "max_tokens", "temperature", "logprobs", "top_logprobs")} == {
        "model": "some-model",
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 20,
    }
    content = body["messages"][0]["content"]
    assert "A. very positive indeed\nB. negative\nC. neither one nor the other" in content
    assert content.endswith("Review: nice\nAnswer:\nAnswer with the letter of one option.")
    assert scorer.device == "api"


def test_rate_limits_and_server_errors_are_retried(api, task, monkeypatch):
    url, state = api
    monkeypatch.setattr(teacher_mod, "API_WAIT", 0.0)
    scorer = ApiScorer(url, "m", task, api_key_env=KEY_ENV, workers=1)
    state["queue"] += [(429, {}), (503, {}), (200, top(("C", -0.5)))]
    assert scorer.score("x") == [-0.5, -0.5, -0.5]
    assert len(state["requests"]) == 3
    state["queue"] += [(500, {})] * teacher_mod.API_ATTEMPTS
    with pytest.raises(SystemExit, match="failed 5 times"):
        scorer.score("x")


def test_the_key_is_sent_only_when_set_and_never_shown(api, task, monkeypatch):
    url, state = api
    monkeypatch.delenv(KEY_ENV, raising=False)
    scorer = ApiScorer(url, "m", task, api_key_env=KEY_ENV, workers=1)
    unkeyed = scorer.fingerprint()
    scorer.score("x")
    assert "Authorization" not in state["requests"][-1][1]
    monkeypatch.setenv(KEY_ENV, SECRET)
    scorer.score("x")
    assert state["requests"][-1][1]["Authorization"] == f"Bearer {SECRET}"
    assert scorer.fingerprint() == unkeyed  # the key does not change the cache
    state["queue"].append((401, {"error": {"message": f"bad key {SECRET}"}}))
    with pytest.raises(SystemExit, match="--api-key-env") as e:
        scorer.score("x")
    assert SECRET not in str(e.value)
    state["queue"].append((400, {"error": {"message": f"bad request {SECRET}"}}))
    with pytest.raises(SystemExit, match="HTTP 400") as e:
        scorer.score("x")
    assert SECRET not in str(e.value)


def test_unsupported_tasks_and_options_are_refused(task):
    multi = Task("m", ["a", "b"], ["a", "b"], "{text} {label}?", multi_label=True)
    with pytest.raises(SystemExit, match="single-label"):
        ApiScorer("http://127.0.0.1:1/v1", "m", multi)
    many = Task("t", [f"l{i}" for i in range(21)], [f"l{i}" for i in range(21)], "{labels}\n{text}")
    with pytest.raises(SystemExit, match="at most 20 labels"):
        ApiScorer("http://127.0.0.1:1/v1", "m", many)
    with pytest.raises(SystemExit, match="not both"):
        load_teacher("m", task, vllm_url="http://a/v1", api_url="http://b/v1")
    with pytest.raises(SystemExit, match="--shortlist"):
        load_teacher("m", task, api_url="http://b/v1", shortlist=2)


def test_lettered_rendering_leaves_the_default_unchanged():
    t = Task(
        "t",
        ["up", "down"],
        ["up", "down"],
        "Pick:\n{labels}\n{examples}\nText: {text}",
        descriptions=[None, "falls"],
    )
    assert t.render("hi") == "Pick:\n- up\n- down: falls\nText: hi"
    assert (
        t.render("hi", [("a  b", 1)]) == 'Pick:\n- up\n- down: falls\n\nExamples:\n- "a b" -> down\nText: hi'
    )
    assert t.render("hi", [("a  b", 1)], lettered=True) == (
        'Pick:\nA. up\nB. down: falls\n\nExamples:\n- "a b" -> B\nText: hi\n'
        "Answer with the letter of one option."
    )


TASK = """\
name: toy
text_field: review
label_field: sentiment
prompt: |
  Classify the sentiment of the review as one of:
  {labels}

  Review: {text}
labels:
  pos: positive review
  neg: negative review
"""


def test_cli_run_with_an_api_teacher(api, tmp_path, monkeypatch, capsys):
    url, state = api
    monkeypatch.setenv(KEY_ENV, SECRET)
    (tmp_path / "task.yaml").write_text(TASK)
    words = ["good", "bad", "great", "awful", "fine", "terrible"]
    rows = [{"review": f"this was {w} {i}"} for i, w in enumerate(words * 2)]
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    test = [{"review": f"so {w}", "sentiment": "pos" if i % 2 == 0 else "neg"} for i, w in enumerate(words)]
    (tmp_path / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in test))
    run = tmp_path / "run"
    main(
        [
            "run",
            str(tmp_path / "task.yaml"),
            "--data",
            str(tmp_path / "train.jsonl"),
            "--test",
            str(tmp_path / "test.jsonl"),
            "--out",
            str(run),
            "--teacher",
            "fake-model",
            "--api-url",
            url,
            "--api-key-env",
            KEY_ENV,
            "--student",
            TINY_ENCODER,
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--reps",
            "2",
            "--teacher-reps",
            "2",
            "--device",
            "cpu",
            "--threads",
            "2",
        ]
    )
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    assert meta["api_url"] == url and meta["api_key_env"] == KEY_ENV and meta["teacher_device"] == "api"
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    teacher = results["models"][0]
    assert teacher["device"] == "api" and teacher["accuracy"] == 1.0 and teacher["latency_ms_b1"] > 0
    assert any(m["name"] == "student, ONNX fp32" for m in results["models"])
    assert all(h.get("Authorization") == f"Bearer {SECRET}" for _, h, _ in state["requests"])
    for f in run.rglob("*"):  # meta.json, the score cache and everything else
        if f.is_file():
            assert SECRET.encode() not in f.read_bytes(), f
    capsys.readouterr()


def test_a_warning_when_the_model_does_not_answer_with_a_letter(api, task):
    url, state = api
    scorer = ApiScorer(url, "chatty-model", task, api_key_env=KEY_ENV, workers=1)
    state["queue"].append((200, top(("The", -0.1), ("Sure", -1.0))))
    with pytest.warns(UserWarning, match="1 of 1 texts: chatty-model's first answer token was not an option"):
        assert scorer.score("nice") == [-1.0, -1.0, -1.0]
    state["queue"].append((200, top(("C", -0.5))))  # one letter, also the smallest value: no warning
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert scorer.score("nice") == [-0.5, -0.5, -0.5]


def test_similar_examples_are_sent_per_text(api):
    from tinydistill.shortlist import Retriever

    url, state = api
    shots = Task("t", ["up", "down"], ["up", "down"], "{labels}\n{examples}\nReview: {text}\nAnswer:")
    pool = [("good film", 0), ("bad film", 1), ("great day", 0)]
    similar = Retriever(pool, 2, TINY_ENCODER)
    scorer = ApiScorer(url, "m", shots, api_key_env=KEY_ENV, workers=1, similar=similar)
    scorer.score_many(["good film", "other"])
    sent = [b["messages"][0]["content"] for _, _, b in state["requests"]]
    (own,) = [c for c in sent if "Review: good film\n" in c]
    shown = own.split("Review:")[0]
    assert shown.count(" -> ") == 2 and '"good film"' not in shown  # never its own example
    assert scorer.fingerprint() != ApiScorer(url, "m", shots, api_key_env=KEY_ENV).fingerprint()
