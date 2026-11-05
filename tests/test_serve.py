import json
import re
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


def test_web_bundle_keeps_labels_first_and_a_size_budget():
    from tinydistill.web import _bundle

    rows = [{"text": "x" * 10, "gold": i // 5} for i in range(20)]  # sorted by label, four labels
    picked, first = _bundle(rows, n=8, chars=1000)
    assert first == 4 and [r["gold"] for r in picked[:4]] == [0, 1, 2, 3] and len(picked) == 8
    assert len({id(r) for r in picked}) == 8  # no row twice
    assert len(_bundle(rows, n=8, chars=35)[0]) == 3  # 10 characters each


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
    (tmp_path / "site" / "model" / "onnx").mkdir(parents=True)  # an older build's folder is replaced
    page = build_site(run / "onnx", tmp_path / "site", title="Demo", note="Data: CC BY 4.0 <x>")
    (folder,) = [p.name for p in (tmp_path / "site").iterdir() if p.is_dir()]
    assert re.fullmatch(r"model-[0-9a-f]{10}", folder)  # named after the model, so browsers never mix copies
    files = {
        p.relative_to(tmp_path / "site").as_posix() for p in (tmp_path / "site").rglob("*") if p.is_file()
    }
    assert {
        "index.html",
        f"{folder}/config.json",
        f"{folder}/tokenizer.json",
        f"{folder}/onnx/model_quantized.onnx",
    } <= files
    assert json.loads((tmp_path / "site" / folder / "config.json").read_text(encoding="utf-8"))[
        "id2label"
    ] == {
        "0": "neg",
        "1": "pos",
    }
    text = page.read_text(encoding="utf-8")
    assert "<title>Demo</title>" in text and '"q8"' in text and "&lt;b&gt;hi&lt;/b&gt;" in text
    assert "<b>hi</b>" not in text and "<b>hi<\\/b>" in text  # escaped in the HTML; "<\/" in the JSON data
    import onnxruntime as ort

    outputs = [
        o.name
        for o in ort.InferenceSession(
            str(tmp_path / "site" / folder / "onnx/model_quantized.onnx")
        ).get_outputs()
    ]
    assert outputs == ["logits", "layer_logits", "cls_attention"]  # the "Inside the model" view
    assert f'"model": "{folder}"' in text
    assert text.count('class="ghost sample"') == 2  # one per gold label, not the first rows
    assert "&lt;/b&gt;&quot;</button>" in text  # the buttons show their own text
    data = json.loads(text.split('id="data">')[1].split("</script>")[0].replace("<\\/", "</"))
    assert len(data["samples"]) == 6 and data["samples"][1]["gold"] == 1  # all test rows, up to BUNDLE
    assert "<td>Test accuracy</td><td class='t'></td><td>50.0%</td>" in text  # the card; no teacher number
    assert 'id="live-ms"' in text and "Size</td>" not in text  # time measured in the page; no sizes: no row
    assert "Test accuracy 0.500 on 6 held-out texts. Distilled from T (8-shot) into S." in text
    assert "Data: CC BY 4.0 &lt;x&gt;" in text
    assert 'class="pages"' not in text and 'class="deeper"' not in text  # no explainer: no tabs to it
    (tmp_path / "site" / "index.html").write_text("the explainer")  # `web` writes it one folder up
    text = build_site(run / "onnx", tmp_path / "site" / "simple", explainer=True).read_text(encoding="utf-8")
    assert (tmp_path / "site" / "index.html").read_text() == "the explainer"
    assert '<a href="./" aria-current="page">Simple</a>' in text and 'data-base="../"' in text
    assert "See inside the model on this text" in text and not re.search(r"__[A-Z_]+__", text)
