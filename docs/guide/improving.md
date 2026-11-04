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

## Target the student's confusions: improve

`improve` runs rounds of performance-guided distillation (PGKD, [Di Palo et al.,
2024](https://arxiv.org/abs/2411.05045)): it finds the label pairs the student confuses most, has the teacher
write texts of each that are easy to mistake for the other, labels them with the run's teacher and trains a new
student with the run's training settings.

```sh
tinydistill improve RUN --rounds 3 --per-round 200
```

The confusions are found on a seeded share of the run's training texts (`--held-out`, default 0.1), held out of
every round's training and compared with the teacher's labels, which are what the student learns. The test set
is never used to choose anything: it is only reported. Round 0 is the run retrained without the held-out texts,
so that every round is compared on texts it did not train on. Each round is a new run directory,
`RUN-improve/round-k`, trained, exported and evaluated like a refresh; the run itself is left as it was.
`improve` stops early when the agreement on the held-out texts does not improve, and writes a table of rounds
(texts added, held-out agreement, test accuracy) to `RUN-improve/improve.md` and `improve.json`. Single-label
runs with a local teacher only; it needs a GPU for real use, since the teacher writes text every round.

## More varied synthetic texts: synth --vary

Texts the teacher writes from the label alone are much alike ([No unlabelled data](no-data.md)). Conditioning
each request on sampled attributes gives more varied texts than the label alone (AttrPrompt, [Yu et al.,
2023](https://arxiv.org/abs/2306.15895)): with `--vary`, each request also names a length, a tone and a writer's
situation, sampled with `--seed`. A task file can replace them with its own `attributes`
([The task file](task-file.md#attributes-for-synthetic-texts)). `--dedup T` drops a text whose cosine similarity
(MiniLM sentence embeddings) to a text kept earlier, or to one in `--data`, is T or more, and logs how many were
dropped:

```sh
tinydistill synth task.yaml --per-label 500 --vary --dedup 0.95 --data real.csv --out texts.jsonl
```

Without the two flags, `synth` writes the same texts as before for a given seed.

## Follow a long run: status

Long steps print their progress, speed and time left, and write them to `RUN/status.json`. From another
terminal, `tinydistill status RUN` shows the same and whether the process is still running.
