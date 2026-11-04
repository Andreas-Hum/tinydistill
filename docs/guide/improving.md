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

### Search over several rounds

With `--rounds R`, each later round starts from the best instruction so far and shows the teacher up to 8 dev
texts it got wrong, each with the correct label and the teacher's answer (long texts cut to 300 characters). The
teacher writes N rewrites meant to fix those mistakes; the same placeholder check applies, and an instruction
already tried is not scored again. The search stops after R rounds, or earlier when a round finds nothing
better.

```sh
tinydistill tune task.yaml --data train.csv --dev-size 400 --few-shot 0 --propose 4 --rounds 4 --out tune
```

To keep the search from fitting the texts it shows, the dev split is cut in two halves (seeded with `--seed`).
Mistakes are shown only from one half, and every candidate, those of round 1 included, is scored only on the
other. So every score in the report, the chosen instruction's too, is on texts the rewrites were not written
for. The teacher still labels the whole split for each candidate, to find its mistakes, so a candidate costs
as much as without rounds. Round 2's candidates are named `tune-r2-1.yaml`, ..., the report gives each
instruction's round, and `tune-best.yaml` is the best of all rounds. A larger `--dev-size` helps here, since
only half of it is scored. Rewriting the prompt from its own mistakes is the idea of GEPA ([Agrawal et al.,
2025](https://arxiv.org/abs/2507.19457)) and of the iterative variant of APE.

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
