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
other task files. Only the instruction is rewritten: the text before the prompt's first placeholder (`{labels}`,
`{text}`, `{examples}` or `{label}`). The rest of the prompt, from there on, is kept as written; a one-word
heading line just before the placeholder, such as `Options:`, is part of it. The teacher is shown the whole
prompt, with the placeholders as plain descriptions such as `[the list of categories]`, and replies with the new
instruction only, so a small teacher has no placeholders to lose. A reply is dropped when it is empty, longer
than 600 characters (or twice the instruction, if more), or has a placeholder or the whole list of label names;
`tune` logs the reason for each, and how many were kept. A prompt that starts with a placeholder has no
instruction to rewrite on its own: the teacher then rewrites the whole prompt, a reply is kept only if it has
every placeholder of the original exactly once and no other, and `tune` says so in the log. A version of the
task with label names only (no descriptions) is always tried when the task has descriptions: on AG News that
alone was the difference in the table above.

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
teacher writes N rewrites of the instruction meant to fix those mistakes; the same checks apply, a reply that
quotes the shown mistakes is dropped too, and an instruction already tried is not scored again. The search
stops after R rounds, or earlier when a round finds nothing better.

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

With a paid API teacher every labelled text costs money. `run --budget N` has the teacher label only N training
texts, picked at random (`--seed`):

```sh
tinydistill run task.yaml --data train.csv --test test.csv --teacher gpt-4o-mini \
  --api-url https://api.openai.com/v1 --budget 2000 --out run
```

With `--rounds R`, the N texts are labelled in R rounds of N/R instead. Round 1 labels N/R texts picked at random
and trains the student. Each later round, the student
scores the texts not labelled yet, the teacher labels the N/R it is least sure of (the highest entropy of its
predicted distribution; multi-label, the mean binary entropy of its labels), and the student is trained again
on everything labelled so far. The last training is the final student, which is exported and evaluated as
usual. This is uncertainty sampling, the classic active-learning rule ([Lewis and Gale,
1994](https://arxiv.org/abs/cmp-lg/9407020)); LLKD ([Li et al., 2024](https://arxiv.org/abs/2411.08028))
also uses the student's uncertainty to choose the texts.

On Banking77 the uncertainty rounds did worse than a random sample of the same size, at every budget tried
([Benchmarks](../results.md#labelling-on-a-budget)): with 1,000 of the 10,003 training texts the student reached
0.383 after 4 rounds against 0.480 at random, and with 5,000, 0.826 against 0.830. The early students, trained on
a few hundred texts, pick ambiguous and unusual texts rather than useful ones. That is why one round, a random
sample, is the default.

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
and evaluates a new student on the same test set. Logged texts that are in the test set are not added:

```sh
tinydistill refresh RUN --log cascade.jsonl --out RUN2
```

The new student trains with RUN's own settings: student, epochs, seeds, learning rate, batch size, gold labels,
`--init-head`, `--trim-vocab`, `--max-length` and int8 export. A training flag given with another value than its
default replaces RUN's, as in `refresh RUN ... --epochs 20`. A trimmed student is trimmed again, to the tokens of
the new training texts, so the logged texts' tokens join its vocabulary.

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

On Banking77, three rounds raised the student's agreement with the teacher on the held-out texts from 0.854 to
0.867 but left its test accuracy where it was (0.856;
[Benchmarks](../results.md#rounds-of-teacher-written-texts-for-the-students-confusions)): the new texts teach the
teacher's view of the confusing pairs, which helps only where the teacher is better than the student.

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
