# Training the student

The student minimises the KL divergence from the teacher's soft labels to its own predictions over the
training texts, plus a cross-entropy term on any gold labels: standard knowledge distillation (Hinton et al.,
2015).

```sh
tinydistill train run --student sentence-transformers/all-MiniLM-L6-v2 --epochs 10 --seeds 3
```

## The student

`--student` takes any Hugging Face encoder. Bigger encoders are more accurate and slower:

| Student | Size (int8) | Latency per text (CPU) |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` (default) | 23 MB | about 1 ms |
| `sentence-transformers/all-MiniLM-L12-v2` | 34 MB | about 3 ms |
| `sentence-transformers/all-mpnet-base-v2` | 110 MB | about 7 ms |
| `answerdotai/ModernBERT-base` | 151 MB | about 10 ms |
| `static` (below) | 8 MB | about 0.1 ms |

On Banking77, MiniLM-L6 and ModernBERT trained for 20 epochs and mpnet trained for 5 reached about the same
accuracy (0.813, 0.815 and 0.814; [Benchmarks](../results.md#banking77-the-hard-case)), so the smallest is the
one to deploy there.

## Long texts

Inputs are cut to `--max-length` tokens (default 128), which suits short messages. For longer texts raise it, and
consider ModernBERT, which reads up to 8,192 tokens; BERT-style students read at most 512. The length is stored
with the model, so `predict`, `serve` and the web page cut inputs the same way.

```sh
tinydistill train run --student answerdotai/ModernBERT-base --max-length 1024
```

## How long to train

`--epochs` is the number of passes over the training texts (default 10). Training longer mattered more than a
bigger student on Banking77: MiniLM-L6 reached 0.729 after 5 epochs and 0.813 after 20, as accurate as mpnet at
a sixth of its latency. On the easier financial-tweets task, 20 epochs gave the same accuracy as 5.

To choose the epoch from the data, `--val-share 0.05` holds out 5% of the training texts (texts without gold
labels, so none are lost), measures the loss on them after every epoch, keeps the best epoch and stops after
`--patience` epochs (default 3) without improvement. No gold labels are needed: the held-out loss is measured
against the teacher's soft labels. Each run writes the loss per epoch to `student/history.json`, and
`tinydistill plot` draws it.

```sh
tinydistill train run --epochs 30 --val-share 0.05
```

!!! tip "Multi-label students need more epochs"
    A multi-label student has a yes/no decision per label to learn. On GoEmotions (28 labels), every seed kept
    its last epoch at 10 epochs and the student reached macro-F1 0.066; with `--epochs 30 --val-share 0.05` it
    reached 0.226, close to its teacher's 0.277.

## Seeds

`--seeds N` trains every student N times with different seeds; the results show the mean and standard deviation,
and the exported model is the first seed's.

## Static student

`--student static` (or `static:<encoder>`) trains a student with no attention at all: one vector per token,
averaged, then a linear layer, started from the encoder's own token representations (as in Model2Vec). It is
much smaller and faster than a transformer student. It ignores word order, so it can miss meaning that depends on
it (negation, "A rather than B"). On Banking77 it reached 0.803, about a point behind MiniLM students trained as
long. It runs with `predict` and `serve` like any other student, but not in the browser page. Train it longer and
at a higher learning rate:

```sh
tinydistill train run --student static --epochs 20 --lr 1e-3
```
