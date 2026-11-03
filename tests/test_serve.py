import json
import threading
import urllib.error
import urllib.request

import pytest
import torch
from conftest import TINY_ENCODER
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tinydistill.export import Predictor, export_onnx
from tinydistill.serve import MAX_TEXTS, make_server


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("serve")
    torch.manual_seed(0)
    model = AutoModelForSequenceClassification.from_pretrained(
        TINY_ENCODER, num_labels=2, id2label={0: "neg", 1: "pos"}, ignore_mismatched_sizes=True
    )
    model.save_pretrained(tmp / "student")
    AutoTokenizer.from_pretrained(TINY_ENCODER).save_pretrained(tmp / "student")
    export_onnx(tmp / "student", tmp / "onnx")
    srv = make_server(Predictor(tmp / "onnx"), port=0)  # any free port
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", Predictor(tmp / "onnx")
    srv.shutdown()


def call(url, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_and_predict(server):
    url, pr = server
    assert call(url + "/health") == (200, {"status": "ok", "labels": ["neg", "pos"]})
    code, body = call(url + "/predict", {"texts": ["good", "bad one"]})
    assert code == 200 and [p["label"] for p in body["predictions"]] == pr.predict(["good", "bad one"])
    assert call(url + "/predict", {"texts": []}) == (200, {"predictions": []})


@pytest.mark.parametrize(
    "raw, code",
    [
        (b"not json", 400),
        (json.dumps({"texts": "one string"}).encode(), 400),
        (json.dumps({"texts": [1, 2]}).encode(), 400),
        (json.dumps({"texts": ["x"] * (MAX_TEXTS + 1)}).encode(), 413),
    ],
)
def test_bad_requests(server, raw, code):
    url, _ = server
    got, body = call(url + "/predict", raw=raw)
    assert got == code and "error" in body


def test_web_page_layout(tmp_path):
    from tinydistill.web import build_site

    run = tmp_path / "run"
    model = AutoModelForSequenceClassification.from_pretrained(
        TINY_ENCODER, num_labels=2, id2label={0: "neg", 1: "pos"}, ignore_mismatched_sizes=True
    )
    model.save_pretrained(run / "student")
    AutoTokenizer.from_pretrained(TINY_ENCODER).save_pretrained(run / "student")
    export_onnx(run / "student", run / "onnx", int8=True)
    rows = [{"text": 'say "<b>hi</b>"', "gold": 0}] + [{"text": f"neg {i}", "gold": 1} for i in range(5)]
    (run / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "results.json").write_text(
        json.dumps(
            {
                "teacher": "T",
                "student": "S",
                "data": {"n_test": 6},
                "models": [
                    {"role": "teacher", "name": "teacher, 8-shot"},
                    {"role": "student", "name": "student, ONNX int8", "accuracy": 0.5},
                ],
            }
        )
    )
    page = build_site(run / "onnx", tmp_path / "site", title="Demo", note="Data: CC BY 4.0 <x>")
    files = {
        p.relative_to(tmp_path / "site").as_posix() for p in (tmp_path / "site").rglob("*") if p.is_file()
    }
    assert {
        "index.html",
        "model/config.json",
        "model/tokenizer.json",
        "model/onnx/model_quantized.onnx",
    } <= files
    assert json.loads((tmp_path / "site/model/config.json").read_text())["id2label"] == {
        "0": "neg",
        "1": "pos",
    }
    text = page.read_text()
    assert (
        "<title>Demo</title>" in text
        and '"q8"' in text
        and "&lt;b&gt;hi&lt;/b&gt;" in text
        and "<b>hi" not in text
    )
    assert text.count('class="ghost sample"') == 2  # one per gold label, not the first rows
    assert "Test accuracy 0.500 on 6 held-out texts. Distilled from T (8-shot) into S." in text
    assert "Data: CC BY 4.0 &lt;x&gt;" in text
