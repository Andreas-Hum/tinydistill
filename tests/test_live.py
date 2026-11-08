import json
import os
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime

import pytest
from conftest import TINY_ENCODER
from test_probe import data  # noqa: F401  (fixture)

from tinydistill.cli import main
from tinydistill.live import serve, state
from tinydistill.progress import Progress


def write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_the_state_reads_the_run_files(tmp_path):
    now = datetime.now(UTC).isoformat(timespec="seconds")
    write(
        tmp_path / "status.json",
        {
            "stage": "train",
            "done": 5,
            "total": 10,
            "unit": "batches",
            "rate_per_s": 2.0,
            "eta_s": 3,
            "finished": False,
            "updated": now,
            "pid": os.getpid(),
        },
    )
    (tmp_path / "status-log.jsonl").write_text(
        json.dumps(
            {"stage": "label train", "done": 4, "total": 4, "unit": "texts", "seconds": 1.0, "finished": now}
        )
        + "\n"
        + '{"stage": "cut off',
        encoding="utf-8",
    )
    write(tmp_path / "history-live.json", {"student, seed 0": [{"epoch": 1, "loss": 0.5}]})
    s = state(tmp_path)
    assert s["status"]["alive"] and s["status"]["stage"] == "train"
    assert [x["stage"] for x in s["stages"]] == ["label train"]  # a line being written is skipped
    assert s["history"]["student, seed 0"][0]["loss"] == 0.5
    assert s["results"] is None
    write(tmp_path / "results.json", {"models": [{"name": "student, PyTorch", "accuracy": 0.9, "extra": 1}]})
    assert state(tmp_path)["results"][0]["accuracy"] == 0.9
    old = time.time() - 3600  # the results of an earlier run in this directory are not this run's
    os.utime(tmp_path / "results.json", (old, old))
    assert state(tmp_path)["results"] is None


def test_an_empty_directory(tmp_path):
    assert state(tmp_path) == {
        "run": str(tmp_path),
        "status": None,
        "stages": [],
        "history": {},
        "results": None,
    }


def test_the_server(tmp_path):
    server = serve(tmp_path, 18765, open_browser=False)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(url + "/") as r:
            assert b"tinydistill live" in r.read()
        with urllib.request.urlopen(url + "/state") as r:
            assert json.loads(r.read())["status"] is None
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(url + "/other")
        second = serve(tmp_path, server.server_address[1], open_browser=False)  # in use: the next port
        assert second.server_address[1] != server.server_address[1]
        second.shutdown()
    finally:
        server.shutdown()


def test_finished_stages_are_logged(tmp_path):
    bar = Progress("label train", 3, lambda _: None, tmp_path / "status.json")
    bar.update(3)
    bar.finish()
    rows = [json.loads(line) for line in (tmp_path / "status-log.jsonl").read_text().splitlines()]
    assert rows[0]["stage"] == "label train" and rows[0]["done"] == 3 and rows[0]["seconds"] >= 0


def test_a_run_writes_what_the_page_reads(tmp_path, data):  # noqa: F811
    run = tmp_path / "run"
    main(
        ["run", *data, "--out", str(run), "--student", TINY_ENCODER, "--epochs", "2", "--seeds", "2"]
        + ["--batch-size", "4", "--reps", "2", "--teacher-reps", "2", "--gold-per-class", "2", "--gold-only"]
    )
    s = state(run)
    assert set(s["history"]) == {
        "student, seed 0",
        "student, seed 1",
        "gold only, seed 0",
        "gold only, seed 1",
    }
    assert [e["epoch"] for e in s["history"]["student, seed 0"]] == [1, 2]
    assert [x["stage"].split(" (")[0] for x in s["stages"]][:3] == ["label train", "label test", "train"]
    assert s["results"] and s["status"]["finished"]
