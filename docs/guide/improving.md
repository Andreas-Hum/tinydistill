# Improving a run

The quickest gains are usually on the teacher's side: a better task file, a stronger teacher, or a few gold
labels. These tools help find them.

## Check the setup first: tune

`tinydistill tune` scores every combination of teacher, number of few-shot examples and task file on a dev split
of gold-labelled *training* texts (never the test set), and ranks them:

```sh
tinydistill tune task.yaml --prompt task-names-only.yaml --data train.csv --dev-size 200 \
  --teacher Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-7B-Instruct --few-shot 0,8 --out tune
```

It writes `tune.md` and `tune.json`, best first, with the `label` command for the winner. On AG News it showed
that label descriptions made the 1.5B teacher collapse onto one label, before any labelling:

--8<-- "examples/ag-news/tune-1.5b.md"

## See where it goes wrong: analyse

`tinydistill analyse RUN` writes `RUN/analysis.html` with each label's scores for teacher and student, the most
common confusions with examples, and texts where the two disagree.

## Label the texts that matter: suggest

`tinydistill suggest RUN --n 200 --out to_label.jsonl` picks the training texts most worth labelling by hand:
where the student and the teacher disagree, then where the teacher is least sure, spread over the labels. Fill in
the empty label field and train again with those labels:

```sh
tinydistill train RUN --gold-file to_label.jsonl
```

## Learn from production: refresh

With a cascade, `predict --cascade RUN --log cascade.jsonl` (and `serve --cascade ... --log`) records every text
passed to the teacher, with the teacher's scores. `refresh` adds them to the training texts and trains, exports
and evaluates a new student on the same test set:

```sh
tinydistill refresh RUN --log cascade.jsonl --out RUN2
```

## Follow a long run: status

Long steps print their progress, speed and time left, and write them to `RUN/status.json`. From another
terminal, `tinydistill status RUN` shows the same and whether the process is still running.
