from tinydistill.teacher import ScoreCache, iter_unique, text_hash


class CountingScorer:
    def __init__(self):
        self.calls = []

    def score_many(self, texts):
        self.calls.extend(texts)
        return [[float(len(t)), -1.0] for t in texts]


def test_resume_scores_only_the_missing_texts(tmp_path):
    path = tmp_path / "cache.jsonl"
    texts = [f"text number {i}" for i in range(5)]
    first = CountingScorer()
    assert ScoreCache(path).fill(first, texts[:3]) == 3

    second = CountingScorer()
    cache = ScoreCache(path)
    assert cache.fill(second, texts) == 2
    assert second.calls == texts[3:]
    assert [cache.get(t) for t in texts] == [[float(len(t)), -1.0] for t in texts]


def test_interrupted_write_is_ignored(tmp_path):
    path = tmp_path / "cache.jsonl"
    ScoreCache(path).fill(CountingScorer(), ["a", "b"])
    with path.open("a") as f:
        f.write('{"h": "' + text_hash("c") + '", "s": [1.0')  # cut off mid-line
    scorer = CountingScorer()
    cache = ScoreCache(path)
    cache.fill(scorer, ["a", "b", "c"])
    assert scorer.calls == ["c"]


def test_a_row_written_after_an_interrupted_one_survives_a_reopen(tmp_path):
    path = tmp_path / "cache.jsonl"
    ScoreCache(path).fill(CountingScorer(), ["a"])
    with path.open("a") as f:
        f.write('{"h": "' + text_hash("b") + '", "s": [1.0')  # cut off mid-line
    ScoreCache(path).fill(CountingScorer(), ["b"])
    scorer = CountingScorer()
    ScoreCache(path).fill(scorer, ["a", "b"])
    assert scorer.calls == []


def test_duplicates_are_scored_once():
    assert list(iter_unique(["x", "y", "x", "z", "y"])) == ["x", "y", "z"]


def test_batches_are_flushed_in_order(tmp_path):
    texts = [f"t{i}" for i in range(10)]
    scorer = CountingScorer()
    cache = ScoreCache(tmp_path / "c.jsonl")
    seen = []
    assert cache.fill(scorer, texts + texts[:3], progress=seen.append, batch=4) == 10
    assert scorer.calls == texts and seen == list(range(1, 11))
    assert len(ScoreCache(tmp_path / "c.jsonl").scores) == 10
