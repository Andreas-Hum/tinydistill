"""The vLLM client against a stand-in server that answers like vLLM's completions endpoint (prompt_logprobs),
computed with a tiny local model."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch
from conftest import TINY_LM
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinydistill.teacher import LabelScorer, VLLMScorer, prompt_members


@pytest.fixture(scope="module")
def fake_vllm():
    model = AutoModelForCausalLM.from_pretrained(TINY_LM).eval()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            choices = []
            for i, prompt in enumerate(body["prompt"]):
                with torch.no_grad():
                    lp = torch.log_softmax(model(torch.tensor([prompt])).logits[0].float(), -1)
                entries = [None] + [
                    {str(t): {"logprob": lp[j - 1, t].item(), "rank": 1, "decoded_token": "?"}}
                    for j, t in enumerate(prompt)
                    if j
                ]
                choices.append({"index": i, "text": "x", "prompt_logprobs": entries})
            data = json.dumps({"choices": choices[::-1]}).encode()  # out of order on purpose
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1", model, requests
    srv.shutdown()


def test_vllm_scores_equal_local_scores(fake_vllm, task):
    url, model, requests = fake_vllm
    tok = AutoTokenizer.from_pretrained(TINY_LM)
    remote = VLLMScorer(url, TINY_LM, tok, task, workers=2)
    local = LabelScorer(model, tok, task)
    texts = ["great product", "meh", ""]
    assert remote.forms == local.forms and remote.prefix_ids == local.prefix_ids
    for got, want in zip(remote.score_many(texts), local.score_many(texts), strict=True):
        assert got == pytest.approx(want, abs=1e-4)
    body = requests[-1]
    assert (
        body["prompt_logprobs"] == 0 and body["model"] == TINY_LM and len(body["prompt"]) == len(local.forms)
    )
    assert remote.fingerprint() != local.fingerprint()  # a server's scores are cached separately


def test_vllm_with_similar_examples_equals_local(fake_vllm, task):
    from conftest import TINY_ENCODER

    from tinydistill.shortlist import Retriever

    url, model, _ = fake_vllm
    tok = AutoTokenizer.from_pretrained(TINY_LM)
    shots = type(task)(**{**task.__dict__, "prompt": "{labels}\n{examples}\n\nReview: {text}\nAnswer:"})
    r = Retriever([("good film", 0), ("bad film", 1), ("so-so", 2)], 2, TINY_ENCODER)
    remote = VLLMScorer(url, TINY_LM, tok, shots, workers=2, similar=r)
    local = LabelScorer(model, tok, shots, similar=r)
    texts = ["great product", "bad film"]
    for got, want in zip(remote.score_many(texts), local.score_many(texts), strict=True):
        assert got == pytest.approx(want, abs=1e-4)
    assert remote.fingerprint() != VLLMScorer(url, TINY_LM, tok, shots).fingerprint()
    sets = Retriever([("good film", 0), ("bad film", 1), ("so-so", 2)], 1, TINY_ENCODER, sets=2)
    remote = prompt_members(VLLMScorer(url, TINY_LM, tok, shots, workers=2, similar=sets), [])
    local = prompt_members(LabelScorer(model, tok, shots, similar=sets), [])
    assert len(remote) == len(local) == 2 and remote[0].heads(texts) != remote[1].heads(texts)
    for r_member, l_member in zip(remote, local, strict=True):
        assert r_member.heads(texts) == l_member.heads(texts)
        for got, want in zip(r_member.score_many(texts), l_member.score_many(texts), strict=True):
            assert got == pytest.approx(want, abs=1e-4)


def test_vllm_reasoning_equals_local_reasoning(fake_vllm, task, monkeypatch):
    from tinydistill.teacher import ReasoningScorer, VLLMReasoningScorer

    url, model, requests = fake_vllm
    tok = AutoTokenizer.from_pretrained(TINY_LM)
    remote = VLLMReasoningScorer(url, TINY_LM, tok, task, 16, workers=2)
    local = ReasoningScorer(model, tok, task, 16)
    monkeypatch.setattr(local, "_generate", lambda prompts: ["x"] * len(prompts))  # what the stand-in writes
    texts = ["great product", "meh"]
    for got, want in zip(remote.score_many(texts), local.score_many(texts), strict=True):
        assert got == pytest.approx(want, abs=1e-4)
    gen = next(b for b in requests if "stop" in b)
    assert gen["max_tokens"] == 16 and gen["temperature"] == 0 and len(gen["prompt"]) == 2
    assert remote.reasoning == {"great product": "x", "meh": "x"}
    assert remote.fingerprint() != VLLMScorer(url, TINY_LM, tok, task).fingerprint()
