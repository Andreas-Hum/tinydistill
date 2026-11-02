from tinydistill.teacher import ScoreCache, iter_unique, text_hash


class CountingScorer:
    def __init__(self):
        self.calls = []

    def score(self, text):
        self.calls.append(text)
        return [float(len(text)), -1.0]


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


def test_duplicates_are_scored_once():
    assert list(iter_unique(["x", "y", "x", "z", "y"])) == ["x", "y", "z"]
