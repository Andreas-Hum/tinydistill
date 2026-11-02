"""The vLLM client against a stand-in server that answers like vLLM's completions endpoint (prompt_logprobs),
computed with a tiny local model."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch
from conftest import TINY_LM
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinydistill.teacher import LabelScorer, VLLMScorer


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
