# Choosing the teacher

The teacher sets the ceiling: the student learns to imitate it, mistakes included. A better teacher usually
helps more than anything on the student side.

## Any causal language model

`--teacher` takes any Hugging Face causal LM with a chat template. The tests check the scoring against a plain
forward pass for Llama, Mistral, Phi, Gemma 2, Gemma 4 and GPT-2 style models as well as Qwen and Qwen3.5.

```sh
tinydistill label task.yaml --data train.csv --teacher Qwen/Qwen2.5-7B-Instruct --load-4bit --out run
```

`--load-4bit` loads it in 4 bit with bitsandbytes (CUDA only, `pip install "tinydistill[4bit]"`): a 7B model
then fits in about 5.5 GB of GPU memory. On Banking77, going from `Qwen2.5-1.5B` to `Qwen2.5-7B` raised the
teacher from 0.404 to 0.629 accuracy, and 8 examples per class in the prompt to 0.783
([Benchmarks](../results.md)).

## Ensembles

Several teachers, comma-separated, are an ensemble: each scores every text (one model in memory at a time) and
the soft labels are the mean of their label probabilities.

```sh
tinydistill run task.yaml --data train.csv --test test.csv --load-4bit \
  --teacher Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3,microsoft/Phi-3.5-mini-instruct
```

`tinydistill compare RUN1 RUN2 ...` puts runs side by side. On financial tweets, an ensemble of three model
families added about a point over the best single teacher.

## Prompt ensembles

`--prompts` takes other task files for the same task, comma-separated: the same label ids in the same order and
the same `multi_label` setting, with another prompt (or other label names or descriptions). Each teacher is
loaded once and scores every text with each prompt, and the soft labels are the mean of all members' label
probabilities, exactly as for several teachers.

```sh
tinydistill label task.yaml --prompts task-b.yaml,task-c.yaml --data train.csv --out run
```

With several teachers the members are every teacher with every prompt. Each member has its own score cache, so
running again with one more prompt scores only that one. The variants are copied into the run directory
(`prompt-1.yaml`, ...) and listed in `meta.json`, so `eval` and the cascade (`predict` and `serve` with
`--cascade`) rebuild the same teacher. Prompt ensembles need a local teacher, not `--vllm-url` or `--api-url`.
Combining the answers of several prompts follows Ask Me Anything ([Arora et al.,
2022](https://arxiv.org/abs/2210.02441)), which combines them with weak supervision; here the probabilities are
averaged.

## Reasoning before answering

`label --reason` has the teacher think briefly before it answers. Reasoning first mainly decides which label
comes out on top, while the shape of the distribution still comes from the model (Chen et al. 2026,
[arXiv 2601.03154](https://arxiv.org/abs/2601.03154)): so the reasoning is for accuracy, and the soft labels are
still read from the label log-probabilities after it.

```sh
tinydistill label task.yaml --data train.csv --teacher Qwen/Qwen2.5-7B-Instruct --reason --out run
```

For each text:

- the prompt ends with an instruction to think it through briefly and then write `Answer:` followed by the
  answer (a trailing `Answer:` in the task's prompt is replaced by it);
- the teacher writes up to `--reason-tokens` tokens (default 256), greedily;
- the reasoning is cut at the first `Answer:`, or, if the model never wrote it, `Answer:` is appended;
- every label is scored as the continuation of prompt + reasoning + `Answer:`, with the usual scoring (all
  surface forms, multi-token labels, `--shortlist`). The scores are the teacher's own log-probabilities.

It is much slower than scoring alone: the reasoning is generated (batched, with left padding) and cannot reuse
the cached part of the prompt. The reasoning of every text is stored with its scores in the cache and written
to `reasoning.jsonl` in the run (one entry per member); `analyse` shows it for the teacher's mistakes.
`--reason` and `--reason-tokens` are part of the cache fingerprint and of `meta.json`, so `eval` and the cascade
use the same teacher. With `--prompts`, every member reasons. It works with local and vLLM teachers (the server
writes the reasoning through its completions endpoint), for single-label tasks; hosted APIs return no
log-probabilities of the answer after the reasoning, so `--api-url` refuses it.

On AG News, reasoning made the 1.5B teacher more accurate (0.778 against 0.755) but not its student (0.785 both
ways; [Benchmarks](../results.md#a-reasoning-step-before-the-answer)). Compare both on a sample (`--limit`) before
paying for reasoning on all your texts.

## A probe on the teacher's hidden states

With gold labels (`--gold-per-class`, `--gold-file`) and a local teacher, `label` fits a probe on them by
default, and the probe alone is the teacher ([below](#fast-labelling-probe-only)). `--probe` also scores every
label and averages the two (slower, a little more accurate; the default with several teachers, `--prompts` or
`--few-shot-sets`), and `--no-probe` turns the probe off, so the teacher scores every label. Both flags are an
error if the probe cannot be fit. The probe fits a linear classifier on the
teacher's own hidden state for the prompt plus each text, and its probabilities are one more ensemble member.
This is prompt-augmented linear probing (PALP; Cho et al., AAAI 2023,
[arXiv 2212.10873](https://arxiv.org/abs/2212.10873)): it matches in-context learning at a few labels per class
and keeps improving as labels are added, where in-context learning levels off.

```sh
tinydistill run task.yaml --data train.csv --test test.csv --gold-per-class 8 --out run
```

- The state is the input of the teacher's LM head at the text's last token: the last layer's normed hidden
  state, from the position that predicts the first label token. The label logits are a linear function of
  exactly this vector. It is recorded during the normal scoring pass (a hook on the LM head, for packed and
  batch-row scoring alike), so there is no extra forward pass and the scores are unchanged.
- `--probe-layer L` reads the output of layer L instead (0 the first, -1 the last, -2 the one before), cached
  in a file of its own. On three tasks a layer about a quarter of the way back was up to 1.6 points better or
  slightly worse than the default, and layers nearer the middle far worse
  ([Benchmarks](../results.md#which-layer-the-probe-reads)), so the default stays; with `--probe-only` a few
  layers can be compared quickly.
- The states are cached next to the scores, as `teacher-<fingerprint>.states.npz` (text hashes and float16
  states). The fingerprint is the same as without the probe. A text whose scores are cached but not its state is
  scored again once, for the state.
- The probe fits on the gold labels `train` uses: `--gold-per-class K` (the same texts, picked with `--seed`)
  and `--gold-file`. A test text is never used. It is multinomial logistic regression (multi-label: one sigmoid
  per label), fit full batch with LBFGS from zero weights, with an L2 penalty `--probe-l2` (default 0.001) on
  standardised states. There are no class weights.
- Its log-probabilities are combined with the in-context scores by the same mean of probabilities as any
  ensemble, and stored as the last entry of `teacher_scores`. `--probe-only` makes the probe the whole teacher
  (below); it needs one teacher, no `--prompts` and no `--few-shot-sets`. With several teachers, the probe reads
  the first teacher's states for the task's own prompt.
- `--probe-dev` (default 0.2) first holds out that share of the gold texts, fits on the rest and logs teacher
  accuracy on the held-out ones: in context, probe alone and combined. The report goes to `meta.json` (`probe`)
  and the results notes. Then the probe is fit again on all the gold texts for labelling.
- The fitted probe is saved as `probe.pt` in the run and its settings in `meta.json`, so `eval` and the cascade
  rebuild the same teacher without fitting again. In an ensemble it reuses the states of its teacher's own pass.

The training rows of the gold texts get the probe's prediction for a text it was fit on, which is more
confident than it would be for a new text. That is fine for distillation, since those rows also have the gold
cross-entropy term, but the teacher's agreement with gold on those rows says nothing about its accuracy: use
the `--probe-dev` report or the test set for that.

With 8 gold examples per class and the 7B teacher shown each text's 16 most similar of them, the probe raised
the teacher's test accuracy from 0.851 to 0.859 on Banking77 and from 0.775 to 0.842 on CLINC150, and the
deployed students from 0.855 to 0.872 and from 0.760 to 0.813; see
[Results](https://github.com/Andreas-Hum/tinydistill/blob/main/RESULTS.md#a-linear-probe-on-the-teachers-hidden-states).

The probe needs a local teacher: hosted APIs and vLLM return no hidden states, so `--api-url` and `--vllm-url`
refuse it. It does not work with `--chunk-tokens`.

### Fast labelling: --probe-only

`--probe-only` is the default with gold labels and a local teacher (not with `--chunk-tokens` or `--reason`).
The teacher scores no label at all: it runs the prompt and each text once, for the state the
probe reads, and the probe's probabilities are the soft labels. Scoring the labels is most of the work, above
all with many labels and for models that cannot pack texts, so this labels many times faster, for a slightly
weaker student:

```sh
tinydistill run task.yaml --data train.csv --test test.csv --teacher google/gemma-4-12B-it --load-4bit \
  --gold-per-class 8 --out run   # with {examples} in the prompt: 16 similar examples and the probe alone
```

On CLINC150 with Gemma 4 12B and each text's 16 most similar examples, the deployed student reached 0.851 after
37 minutes of labelling, against 0.859 after 7.2 hours when the labels are scored too. With a fixed zero-shot
prompt (no `--few-shot`), encoded once for every text, labelling Banking77 took 74 seconds for a student of
0.824 ([Benchmarks](../results.md#the-probe-alone-minutes-of-labelling-instead-of-hours)). Its caches have a
fingerprint of their own, and `--probe-dev` reports the probe's accuracy alone.

## A vLLM server

A vLLM server can be the teacher instead of a local model, so a large model on a GPU cluster does the
labelling. Only the tokenizer is loaded locally; every continuation is scored exactly from the server's prompt
log-probabilities, so the soft labels are the same as with the local model (checked on 300 texts: the same top
label for all of them).

```sh
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-prefix-caching --host 127.0.0.1
tinydistill label task.yaml --data train.csv --teacher Qwen/Qwen2.5-7B-Instruct \
  --vllm-url http://127.0.0.1:8000/v1 --out run
```

## Hosted APIs

A hosted OpenAI-compatible chat API (OpenAI, DeepInfra, Together, a local llama.cpp server, ...) can be the
teacher too. `--api-url` is the API's base URL and `--teacher` the model's name there; the API key is read from
the environment variable named by `--api-key-env` (default `OPENAI_API_KEY`). If that variable is unset, no key
is sent, as a local server needs none. Only the variable's name is written to `meta.json`; the key never
appears in the run directory, the cache or error messages.

```sh
export OPENAI_API_KEY=...
tinydistill label task.yaml --data train.csv --teacher gpt-4o-mini \
  --api-url https://api.openai.com/v1 --out run
```

Such APIs return no log-probabilities for the prompt, only the top 20 for the first generated token, so the
exact scoring of every label name is not possible. Instead the labels are shown as lettered options (`A. name`,
`B. name: description`, ...), few-shot examples answer with the letter, the message ends with "Answer with the
letter of one option.", and each label's score is the log-probability of its letter as the first answer token
(`A`, ` A` and `a` all count). A letter is one token in every tokenizer, so this is exact, but:

- only single-label tasks with at most 20 labels work; for more labels or multi-label tasks use a local or
  vLLM teacher (`--shortlist` is for local teachers only);
- a letter that is not among the 20 returned tokens gets the smallest returned log-probability, an upper
  bound on its true value;
- the prompt should not ask for the label name, or the model may not answer with a letter at all.

Rate limits (HTTP 429), server errors and timeouts are retried with back-off. `eval` times the teacher by
calling the API again on the test texts; `--skip-teacher-bench` avoids that.

Every text is sent to the API provider: check their data policy before labelling anything personal or
confidential.

### What will it cost: --dry-run

`--dry-run` prints what labelling would take, and scores nothing: the texts not in the cache yet (training and
test), the requests and the input and output tokens. With the API's prices per million tokens, `--price-in`
and `--price-out`, it also prints the cost. There is no built-in price list; prices change.

```sh
tinydistill label task.yaml --data train.csv --teacher gpt-4o-mini \
  --api-url https://api.openai.com/v1 --dry-run --price-in 0.15 --price-out 0.6 --out run
```

The input tokens are an approximation: the characters of each rendered prompt divided by 4 (a hosted model's
tokenizer is not at hand), with the few-shot examples it would show. The output tokens are the most the
requests ask for: one token per text, or with `--api-samples N` 10 tokens per answer. With `--api-samples N`
the estimate counts N requests per text, each with the whole prompt; an API that accepts `n` takes one request
and bills the prompt once. To label only part of the data, see `run --budget` in
[Improving a run](improving.md#label-on-a-budget-run-budget).

### APIs without log-probabilities

Some APIs and models return no log-probabilities at all. For them, `--api-samples N` (N at least 2) asks for N
answers per text at temperature `--api-temperature` (default 0.7), and each label's soft label is its share of
the answers, in the spirit of self-consistency (Wang et al. 2022,
[arXiv 2203.11171](https://arxiv.org/abs/2203.11171)). There is no smoothing: a label no answer chose gets
probability (practically) 0.

```sh
tinydistill label task.yaml --data train.csv --teacher some-model \
  --api-url https://api.example.com/v1 --api-samples 10 --out run
```

- It costs N times the requests: N answers per text, in one request with `n: N` where the API supports it,
  otherwise in N requests.
- The soft labels are coarse with small N: 5 answers give probabilities in steps of 1/5. Use the
  log-probability scoring above whenever the API offers it.
- An answer counts when it begins with an option letter standing alone: `B`, `B. negative`, `(B)`, `**B**`,
  `Answer: B` and `The answer is B` all count for B, and case is ignored. A letter followed by a word does not
  count, so "A dog barked" is not option A. Answers without a letter are left out; if none of a text's answers
  has one, every label gets the same score and the run warns.
- N and the temperature are part of the cache fingerprint and are written to `meta.json`, so `eval` and
  `predict --cascade` use the same teacher.

## Labelling speed

The part of the prompt before the text is run once and cached; each text then needs one forward pass that
covers all labels together, and short texts are packed several to a pass (`--teacher-tokens`, default 2048
tokens per pass). Models with linear-attention or sliding-window layers (such as Qwen3.5 and Gemma 2 to 4)
cannot be packed: each text and each label continuation is a batch row of its own instead, texts of the same
token length run together, and each row holds its own copy of the cached prompt. A pass takes as many rows as
fit in a quarter of the free GPU memory (2 GiB on CPU) and `--teacher-tokens`. Scores are cached by text in
`--cache-dir`: an interrupted run resumes, runs that share the directory label each text only once, and a run
whose texts are all cached does not load the teacher at all.
With very many labels, see [Many labels](many-labels.md). For documents longer than the student reads,
`--chunk-tokens` has the teacher score windows of each one; see [Long texts](training.md#long-texts).

### What the cache holds: tinydistill cache

Caches grow: a run with the probe also keeps a state per text, and every setup (teacher, prompt, examples,
shortlist) has a cache of its own. `tinydistill cache runs/cache` lists them, largest first: the texts, the size,
when each was last written, and which teacher and runs use it (from the `meta.json` of the runs below `--runs`,
by default the cache directory's parent). `--prune` deletes the caches no run uses, except those written in the
last 24 hours (`--keep-recent H`), since a run still labelling writes its `meta.json` only when it is done.

### How long will it take: --dry-run

With a local or vLLM teacher, `--dry-run` loads it, scores 8 texts to warm up (the prompt's cache, the GPU's
kernels), times up to 64 more of the texts not in the cache yet and prints how long labelling all of them would
take; with an ensemble, a line per member and a total. `run --dry-run` also times a few training batches of the
student and prints how long training every seed would take. Nothing is cached or written. Labelling times vary a lot
with the setup: on Banking77, 74 seconds for the 7B teacher's hidden states alone with a fixed prompt
(`--probe-only`), and 35 minutes for the same teacher scoring every label with each text's 16 most similar
examples ([Benchmarks](../results.md)).

```sh
tinydistill label task.yaml --data train.csv --test test.csv --teacher Qwen/Qwen2.5-7B-Instruct --load-4bit \
  --few-shot-similar 16 --gold-per-class 8 --dry-run --out run
```

## Few-shot examples

`label --few-shot K` shows the teacher K gold examples per class, the same for every text;
`label --few-shot-similar M` shows each text the M gold examples most similar to it (Liu et al., 2021). See
[A few gold labels](gold-labels.md#in-the-teachers-prompt).

With `--gold-per-class K` and neither option given, each text's prompt shows its 16 most similar of those K
gold examples per class, the setting of the best runs on every benchmark task, when the prompt has an
`{examples}` line before `{text}` (as the one `tinydistill init` writes) and the task is single-label;
`--few-shot-similar 0` shows none.

### Several sets of similar examples

`label --few-shot-similar M --few-shot-sets K` scores each text K times. Set k shows it ranks k·M+1 to (k+1)·M
of its most similar gold examples, ordered most similar last as with one set, and the soft labels are the mean
of the K sets' label probabilities. Splitting the demonstrations into buckets, scoring each bucket's prompt
separately and combining the predictions is demonstration ensembling ([Khalifa et al.,
2023](https://arxiv.org/abs/2308.08780)).

```sh
tinydistill label task.yaml --data train.csv --few-shot-similar 8 --few-shot-sets 3 --out run
```

- Set 0 shows the same examples as `--few-shot-similar M` alone, and later sets show less similar ones. The
  sets are consecutive ranks rather than a random split of the top K·M, so each set's examples are fixed by the
  data and no seed enters the prompt.
- Each set is a member of the teacher like a prompt in a [prompt ensemble](#prompt-ensembles): one model in
  memory, its own score cache, its scores in `teacher_scores` (which `analyse` reads). With `--prompts` the
  members are every prompt with every set; with several teachers, every teacher with each of those.
- Labelling takes about K times as long as with one set.
- With 8 gold examples per class, three sets of 16 were worse than one on Banking77 and CLINC150, for the
  teacher and the student; see [Results](https://github.com/Andreas-Hum/tinydistill/blob/main/RESULTS.md#several-sets-of-similar-examples-worse-than-one).
  The later sets' examples are less similar to the text, so the option is more likely to help with more gold
  examples per class to draw from.
- It needs `--few-shot-similar`; with only `--few-shot`, every text sees the same examples, and sets of them
  are not offered. K·M must be smaller than the number of gold examples to pick from, since a training text is
  never its own example; `label` stops otherwise.
- It works with local, vLLM and hosted API teachers. `meta.json` records `few_shot_sets`, so `eval`, the
  cascade, `refresh` and `improve` rebuild the same teacher. One set (the default) keeps the fingerprint and
  cache of `--few-shot-similar` alone.

## Compare setups before labelling

`tinydistill tune` scores combinations of teacher, few-shot examples and prompt on a few hundred gold-labelled
training texts and ranks them; see [Improving a run](improving.md#check-the-setup-first-tune).
