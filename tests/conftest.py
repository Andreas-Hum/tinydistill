import os

# Tests run on CPU only, with few threads.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import pytest  # noqa: E402

from tinydistill.task import Task  # noqa: E402

TINY_LM = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_ENCODER = "hf-internal-testing/tiny-random-BertForSequenceClassification"


@pytest.fixture
def task() -> Task:
    return Task(
        name="toy",
        labels=["pos", "neg", "neutral"],
        verbalisations=["very positive indeed", "negative", "neither one nor the other"],
        prompt="Is this review positive, negative or neutral?\n{labels}\n\nReview: {text}\nAnswer:",
    )
