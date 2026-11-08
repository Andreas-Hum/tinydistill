# A few gold labels

tinydistill needs no gold labels, but if some training texts have them, they help in three places. All use the
same K examples per class, chosen at random with `--seed`.

## In the teacher's prompt

`label --few-shot K` shows K gold examples per class in the teacher's prompt, in place of the `{examples}` line.
The prompt prefix is computed once, so even hundreds of examples cost little per text.

```sh
tinydistill run task.yaml --data train.csv --test test.csv --teacher Qwen/Qwen2.5-7B-Instruct --load-4bit \
  --few-shot 8 --gold-per-class 8 --gold-only --out run
```

On Banking77, 8 examples per class raised the 7B teacher from 0.629 to 0.783 accuracy.

!!! warning "Few-shot examples can hurt"
    K examples per class tell the teacher that the classes are equally common, and the last examples in the
    prompt pull it towards their labels. On the skewed financial-tweets task the 8-shot 1.5B teacher was much
    worse than the zero-shot one. Compare both before relying on it (`tinydistill tune` or
    `tinydistill compare`).

### Examples chosen per text

`label --few-shot-similar M` shows each text its own examples: the M gold training examples most similar to it.
Choosing in-context examples by similarity to the input rather than at random is the method of
[Liu et al. (2021)](https://arxiv.org/abs/2101.06804). Similarity is the cosine of sentence embeddings from the
same small model as `--shortlist` (MiniLM; `--few-shot-model` picks another). The examples are ordered from
least to most similar, so the most similar one is next to the text. A text is never shown as its own example.

```sh
tinydistill label task.yaml --data train.csv --teacher Qwen/Qwen2.5-7B-Instruct --few-shot-similar 8 --out run
```

- Alone, it picks from every gold-labelled training text. With `--few-shot K` it picks from those K examples
  per class, so the teacher sees no other gold labels.
- The prompt needs `{examples}` before `{text}`. Only the part before `{examples}` is computed once; the
  examples are part of every text's tokens, so labelling is slower than with the same examples for all texts.
- It cannot be used with `--chunk-tokens`: a window is not its document's text, so the document would be one
  of its examples.
- It works with local, vLLM and hosted API teachers. The examples it picks from are saved in the run
  (`examples.jsonl`), so `eval` and the cascade rebuild the same teacher.

## In a probe on the teacher

With a local teacher, `label` (and `run`) fits a linear probe on the teacher's hidden states with the gold
labels of `--gold-per-class` and `--gold-file`, and averages its probabilities with the teacher's; it is on by
default whenever there are gold labels, and `--no-probe` turns it off. See
[Choosing the teacher](teacher.md#a-probe-on-the-teachers-hidden-states).

## In the student's loss

`train --gold-per-class K` adds a cross-entropy term on those gold labels to the distillation loss (weight
`--ce-weight`, default 1). `train --gold-file FILE` does the same with labels from a file, such as the texts
[`suggest`](improving.md#label-the-texts-that-matter-suggest) picked.

## As baselines

- `train --gold-only` also trains the same student on those labels alone, without the teacher: what the
  teacher adds over the few labels.
- `train --upper-bound` trains it on all gold training labels: how far distillation is from full supervision.

Both appear as extra rows in the results. For a stronger few-label baseline, the benchmarks compare with
[SetFit](../results.md#compared-with-setfit) on the same labels.

## Adding a label later

A new label needs a new teacher pass: the prompt lists the labels, so the teacher's cached scores and states
belong to the old label set. With the probe that pass can be short:

1. Add the label to the task file, after the existing ones.
2. Find a few training texts that belong to it and give them the new label in a `--gold-file` (the same format
   as the training data; its labels apply to those training texts). Along with `--gold-per-class` for the
   other labels, the probe learns the new label from these examples.
3. Run again with `--probe-only`. The teacher then scores no label, it only gives the probe its hidden states:
   on Banking77 that took 74 seconds with a zero-shot prompt, and about 10 minutes on CLINC150
   ([Benchmarks](../results.md#the-probe-alone-minutes-of-labelling-instead-of-hours)).

```sh
tinydistill run task.yaml --data train.csv --test test.csv --gold-per-class 8 --gold-file new-label.csv \
  --probe-only --out run-with-new-label
```

Without `--probe-only` the teacher scores every label again, as in the first run, which takes longer and is a
little more accurate.
