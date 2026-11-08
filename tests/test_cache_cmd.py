import json
import os
import time

from tinydistill.cli import main


def test_the_cache_command_lists_and_prunes(tmp_path, capsys):
    cache, runs = tmp_path / "cache", tmp_path
    cache.mkdir()
    for fp in ("used", "old", "recent"):
        (cache / f"teacher-{fp}.jsonl").write_text('{"h": "a", "s": [0]}\n{"h": "b", "s": [0]}\n')
    (cache / "teacher-used.states.npz").write_bytes(b"x" * 10)
    (cache / "teacher-old.states-layer3.npz").write_bytes(b"x")
    (cache / "teacher-keys.json").write_text(json.dumps({"q1": "used", "q2": "old"}))
    old = time.time() - 3 * 86400
    for f in cache.glob("teacher-old*"):
        os.utime(f, (old, old))
    (runs / "group" / "run").mkdir(parents=True)
    (runs / "group" / "run" / "meta.json").write_text(
        json.dumps(
            {
                "teacher": "org/model-a,org/model-b",
                "teachers": ["org/model-a", "org/model-b"],
                "fingerprints": ["other", "used"],
            }
        )
    )
    main(["cache", str(cache)])
    out = capsys.readouterr().out
    rows = {line.split()[0]: line for line in out.splitlines() if line.startswith(("used", "old", "recent"))}
    assert "model-b" in rows["used"] and "group/run" in rows["used"] and "2" in rows["used"].split()[1]
    assert "no run" in rows["old"] and "no run" in rows["recent"]
    assert "1 of them written in the last 24 hours, kept" in out and "--prune` deletes the other 1" in out
    assert len(list(cache.glob("teacher-old*"))) == 2  # listing deletes nothing
    main(["cache", str(cache), "--prune"])
    assert "deleted 1 caches" in capsys.readouterr().out
    assert not list(cache.glob("teacher-old*"))
    assert (cache / "teacher-used.states.npz").exists() and (cache / "teacher-recent.jsonl").exists()
    assert json.loads((cache / "teacher-keys.json").read_text()) == {"q1": "used"}
