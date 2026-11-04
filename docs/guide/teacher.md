# Choosing the teacher

The teacher sets the ceiling: the student learns to imitate it, mistakes included. A better teacher usually
helps more than anything on the student side.

## Any causal language model

`--teacher` takes any Hugging Face causal LM with a chat template. The tests check the scoring against a plain
forward pass for Llama, Mistral, Phi, Gemma 2 and GPT-2 style models as well as Qwen.

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

## Labelling speed

The part of the prompt before the text is run once and cached; each text then needs one forward pass that
covers all labels together, and short texts are packed several to a pass (`--teacher-tokens`, default 2048
tokens per pass). Scores are cached by text in `--cache-dir`: an interrupted run resumes, runs that share the
directory label each text only once, and a run whose texts are all cached does not load the teacher at all.
With very many labels, see [Many labels](many-labels.md). For documents longer than the student reads,
`--chunk-tokens` has the teacher score windows of each one; see [Long texts](training.md#long-texts).

## Compare setups before labelling

`tinydistill tune` scores combinations of teacher, few-shot examples and prompt on a few hundred gold-labelled
training texts and ranks them; see [Improving a run](improving.md#check-the-setup-first-tune).
