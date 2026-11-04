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

## Labelling speed

The part of the prompt before the text is run once and cached; each text then needs one forward pass that
covers all labels together, and short texts are packed several to a pass (`--teacher-tokens`, default 2048
tokens per pass). Scores are cached by text in `--cache-dir`: an interrupted run resumes, runs that share the
directory label each text only once, and a run whose texts are all cached does not load the teacher at all.
With very many labels, see [Many labels](many-labels.md).

## Compare setups before labelling

`tinydistill tune` scores combinations of teacher, few-shot examples and prompt on a few hundred gold-labelled
training texts and ranks them; see [Improving a run](improving.md#check-the-setup-first-tune).
