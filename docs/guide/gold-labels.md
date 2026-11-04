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
