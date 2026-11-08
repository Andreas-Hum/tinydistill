from conftest import TINY_ENCODER
from test_probe import data  # noqa: F401  (fixture)

from tinydistill.cli import main
from tinydistill.student import seconds_per_batch


def files(path):
    return sorted(p.relative_to(path) for p in path.rglob("*")) if path.exists() else []


def test_a_local_teacher_dry_run_times_a_sample_and_writes_nothing(tmp_path, data, capsys):  # noqa: F811
    run, cache = tmp_path / "run", tmp_path / "cache"
    main(["label", *data, "--out", str(run), "--dry-run"])
    out = capsys.readouterr().out
    assert "28 texts to label, about " in out and "texts/s (timed on 20 of them)" in out
    assert "Nothing was cached or written" in out
    assert files(run) == [] and files(cache) == []
    main(["label", *data, "--out", str(run), "--limit", "4"])  # 4 training and the 4 test texts labelled
    before = {p: (run / p).read_bytes() for p in files(run) if (run / p).is_file()}
    cached = files(cache)
    main(["label", *data, "--out", str(run), "--dry-run"])
    assert "20 texts to label" in capsys.readouterr().out
    assert {p: (run / p).read_bytes() for p in files(run) if (run / p).is_file()} == before
    assert files(cache) == cached


def test_run_dry_run_also_times_training(tmp_path, data, capsys):  # noqa: F811
    run = tmp_path / "run"
    main(
        ["run", *data, "--out", str(run), "--dry-run", "--student", TINY_ENCODER, "--epochs", "2"]
        + ["--batch-size", "4", "--seeds", "2"]
    )
    out = capsys.readouterr().out
    assert "texts to label, about " in out
    assert f"training: 24 batches of {TINY_ENCODER} on cpu (2 seeds), about " in out  # 2 x 2 x 24/4
    assert files(run) == []


def test_seconds_per_batch():
    assert (
        0 < seconds_per_batch(TINY_ENCODER, ["a", "b"], ["some text", "more"], "cpu", batch_size=2, n=2) < 60
    )
