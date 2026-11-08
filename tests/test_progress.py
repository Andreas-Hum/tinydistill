import json
import os

from tinydistill.progress import Progress, describe_status, duration


def test_duration():
    assert [duration(s) for s in (5, 75, 3725)] == ["5s", "1m15s", "1h02m"]


def test_progress_lines_and_status_file(tmp_path):
    lines = []
    status = tmp_path / "status.json"
    p = Progress("label train", 10, lines.append, status, every=0, write_every=0, bar=False)
    for _ in range(4):
        p.update()
    assert lines[-1].startswith("label train: 4/10 (40%) texts,") and "left" in lines[-1]
    s = json.loads(status.read_text(encoding="utf-8"))
    assert s["done"] == 4 and s["total"] == 10 and s["pid"] == os.getpid() and not s["finished"]
    assert "running" in describe_status(status) and "left" in describe_status(status)
    p.finish()
    assert "done in" in lines[-1] and json.loads(status.read_text(encoding="utf-8"))["finished"]
    assert "finished" in describe_status(status)


def test_status_of_a_stopped_run(tmp_path):
    status = tmp_path / "status.json"
    Progress("train", 100, lambda m: None, status)
    record = json.loads(status.read_text(encoding="utf-8")) | {
        "pid": 2**22 + 12345
    }  # a process that does not exist
    status.write_text(json.dumps(record))
    assert "not running" in describe_status(status)
    assert "no status yet" in describe_status(tmp_path / "missing.json")


def test_unknown_total():
    lines = []
    p = Progress("label", None, lines.append, every=0)
    p.update(3)
    assert lines[-1].startswith("label: 3 texts,") and "left" not in lines[-1]
