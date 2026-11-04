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

### Let the teacher propose instructions

`--propose N` has the (first) teacher write N rewrites of the task's instruction and scores them alongside the
other task files. A rewrite is kept only if it has every placeholder of the original (`{labels}`, `{text}`,
`{examples}` or `{label}`) exactly once and no other; `tune` logs how many were dropped. A version of the task
with label names only (no descriptions) is always tried when the task has descriptions: on AG News that alone
was the difference in the table above.

```sh
tinydistill tune task.yaml --data train.csv --dev-size 200 --few-shot 0 --propose 8 --out tune
```

Each candidate is written as a task file next to the report (`tune-1.yaml`, ..., `tune-names-only.yaml`), the
best one as `tune-best.yaml`, and `tune.md` lists every instruction with its dev accuracy. The rewrites are
sampled with `--seed`, so a rerun on the same machine proposes the same ones. Searching over instructions the
model writes itself, scored on labelled examples, is the idea of APE ([Zhou et al.,
2022](https://arxiv.org/abs/2211.01910)). The dev split is small, so a rewrite that wins by a few texts may not
be better; check the winner on more data before relying on it.

## See where it goes wrong: analyse

`tinydistill analyse RUN` writes `RUN/analysis.html` with each label's scores for teacher and student, the most
common confusions with examples, and texts where the two disagree.

### Check the gold labels: analyse --review

Gold labels have mistakes too. `tinydistill analyse RUN --review review.csv` writes, instead of the page, a CSV
of training texts for a person to check (single-label runs; no student or test set needed):

- first the texts whose gold label the teacher's top label contradicts, the teacher's most confident answer
  first: where a confident teacher disagrees, the gold label is the likeliest to be wrong (or the prompt
  misleads the teacher);
- then, for an ensemble of teachers or prompts, the other texts on which its members' top labels differ, the
  most divided first.

The columns are `text`, `gold`, `teacher_label`, `teacher_confidence` and `member_labels`. This is the "find
label issues" half of confident learning ([Northcutt et al.,
2021](https://arxiv.org/abs/1911.00068)), without the other half: training is not changed. Fix the labels you
find wrong in the data and label again.

## Label the texts that matter: suggest

`tinydistill suggest RUN --n 200 --out to_label.jsonl` picks the training texts most worth labelling by hand:
where the student and the teacher disagree, then where the teacher is least sure, spread over the labels. Fill in
the empty label field and train again with those labels:

```sh
tinydistill train RUN --gold-file to_label.jsonl
```

## Label on a budget: run --budget

With a paid API teacher every labelled text costs money. `run --budget N --rounds R` has the teacher label only
N training texts, in R rounds (default 4) of N/R:

```sh
tinydistill run task.yaml --data train.csv --test test.csv --teacher gpt-4o-mini \
  --api-url https://api.openai.com/v1 --budget 2000 --rounds 4 --out run
```

Round 1 labels N/R texts picked at random (`--seed`) and trains the student. Each later round, the student
scores the texts not labelled yet, the teacher labels the N/R it is least sure of (the highest entropy of its
predicted distribution; multi-label, the mean binary entropy of its labels), and the student is trained again
on everything labelled so far. The last training is the final student, which is exported and evaluated as
usual. This is uncertainty sampling, the classic active-learning rule ([Lewis and Gale,
1994](https://arxiv.org/abs/cmp-lg/9407020)); LLKD ([Li et al., 2024](https://arxiv.org/abs/2411.08028))
also uses the student's uncertainty to choose the texts.

- The rounds only choose which texts the teacher labels; the student is trained on them exactly as without a
  budget.
- `train.jsonl` holds the N labelled texts only, so `--gold-per-class` and `--upper-bound` see only those.
- The `--test` texts are labelled too, outside the budget; leave out `--test` or use `--test-limit` to save.
- The scores are cached as usual, so a larger budget later pays only for the new texts.
- `run --budget N --dry-run` estimates the cost of N texts picked at random (see
  [Hosted APIs](teacher.md#hosted-apis)).

Without `--budget` nothing changes.

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
