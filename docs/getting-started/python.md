# From Python

## In a few lines: Distiller

`Distiller` runs the whole pipeline scikit-learn style: `fit` scores the texts with the teacher, trains the
student on its soft labels and exports it to ONNX, and the fitted model predicts.

```python
from tinydistill import Distiller, load_task, read_examples

task = load_task("task.yaml")
texts = [e.text for e in read_examples("hf:PolyAI/banking77:train", task, limit=1000)]
test = list(read_examples("hf:PolyAI/banking77:test", task, limit=500))

clf = Distiller(task, teacher="Qwen/Qwen2.5-1.5B-Instruct", cache_dir="cache")
clf.fit(texts)  # optionally fit(texts, gold) with a gold label (or None) per text
clf.predict(["How do I top up my card?"])  # one label id per text
clf.predict_proba(["How do I top up my card?"])  # numpy array, one column per label in clf.classes_
clf.score([e.text for e in test], [e.gold for e in test])  # accuracy
clf.save("model")  # a folder Predictor reads

clf = Distiller.load("model")  # no teacher needed
```

`cache_dir` keeps the teacher's scores in the same files `tinydistill label --cache-dir` writes, so a second
`fit` on the same texts (or a command-line run) does not load the teacher again. `get_params` and `set_params`
return and set the constructor's arguments, so scikit-learn tools such as `clone` work with it.

With gold labels and a task prompt with an `{examples}` line, `fit(texts, gold)` shows the teacher each text's 16
most similar gold-labelled texts as examples, as `tinydistill label` does with `--gold-per-class`;
`few_shot_similar=0` turns that off, and `few_shot=K` picks K per class to show (or to pick the similar ones
from). With gold labels, `fit(texts, gold)` also fits a linear probe on the teacher's hidden states with them,
and the probe alone is the teacher, as `tinydistill label` does by default
([Choosing the teacher](../guide/teacher.md#a-probe-on-the-teachers-hidden-states)): the teacher scores no label,
it only gives the probe its hidden states, so labelling takes a fraction of the time. `probe_only=False` also
scores every label and averages the two (slower, a little more accurate); `probe=False` turns the probe off:

```python
clf = Distiller(task, teacher="Qwen/Qwen2.5-7B-Instruct", load_4bit=True, cache_dir="cache")
clf.fit(texts, gold)  # gold: a label for a few texts per class, None for the rest
```

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
