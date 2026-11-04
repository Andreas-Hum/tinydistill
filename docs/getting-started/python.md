# From Python

## Use a distilled model

`Predictor` runs an exported model with onnxruntime and a tokenizer; it does not need PyTorch.

```python
from tinydistill import Predictor

clf = Predictor("run/onnx")  # uses model.int8.onnx when present
clf = Predictor("hf:Shroomy/tinydistill-banking77-minilm")  # or a model on the Hugging Face Hub

clf.predict(["How do I top up my card?"])  # one label name per text
clf.predict_proba(["How do I top up my card?"])  # numpy array, one row of probabilities per text
clf.describe(["How do I top up my card?"])  # [{'label': ..., 'confidence': ...}]
```

For a multi-label model, `predict` returns the list of labels with probability 0.5 or more for each text, and
`describe` every label's probability.

!!! note "int8 runs one text at a time"
    The int8 model's dynamic quantisation scales activations over the whole batch, so in a batch a text's
    scores would depend on the other texts in it. `Predictor` therefore runs it one text at a time by default;
    on a CPU this costs little throughput. Pass `batch_size` to override.

## The whole pipeline

Everything the command line does is available as functions:

```python
from pathlib import Path

import torch
from tinydistill import Predictor, export_onnx, load_task, load_teacher, read_examples, train_student

task = load_task("task.yaml")
texts = [e.text for e in read_examples("hf:PolyAI/banking77:train", task, limit=1000)]

teacher = load_teacher("Qwen/Qwen2.5-1.5B-Instruct", task, device="cuda")
soft_labels = torch.tensor(teacher.score_many(texts)).softmax(-1)  # one row of label scores per text

train_student(texts, soft_labels, task.labels, Path("student"), epochs=10)
export_onnx(Path("student"), Path("student-onnx"), int8=True)
Predictor("student-onnx").predict(["Where is my card?"])
```

## Charts

With the `plot` extra, every chart of a run is a function that returns a matplotlib figure, so it shows inline
in a notebook:

```python
from tinydistill import plots

plots.soft_labels("run")  # the teacher's probabilities next to the student's
plots.speed_accuracy("run")  # accuracy against latency
plots.training("run")  # loss per epoch
```

See [Results and charts](../guide/evaluation.md) for all of them, and the [Python API](../reference/python.md)
for every function.
