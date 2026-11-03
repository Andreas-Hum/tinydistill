# Changelog

## Unreleased

- `label` does not load the teacher when every text is already in the cache, so training new students on
  cached labels needs neither the time nor the GPU for the teacher.

## 0.3.0 (2026-11-03)

- Few labels: `label --few-shot K` puts K gold examples per class in the teacher's prompt (`{examples}`);
  `train --gold-only` adds a baseline student trained on those labels alone; `train --seeds N` reports the mean
  and standard deviation over training seeds.
- The teacher packs several short texts into one forward pass (`--teacher-tokens`).
- New commands: `init` (a starter task file from a dataset), `compare` (one table across runs) and `push`
  (upload to the Hugging Face Hub with a model card).
- The package exports the whole pipeline for use from Python.
- Results tables list the baseline and upper-bound rows after the deployable rows.
- Teachers: any causal LM (tested for Llama, Mistral, Phi, Gemma 2, GPT-2 and Qwen), ensembles of several
  (`--teacher a,b`), and remote teachers on a vLLM server (`--vllm-url`).
- Fix: scoring was wrong for models that use eager attention (such as Gemma 2); the mask is now additive.
- The prompt prefix is encoded in chunks, so long few-shot prompts no longer need gigabytes for attention.
- Cascade: `eval` reports accuracy and latency when unsure texts go to the teacher; `predict --cascade` and
  `serve --cascade` route them at inference.
- New commands: `serve` (JSON HTTP endpoint), `web` (a static page running the model in the browser with
  transformers.js) and `synth` (the teacher writes unlabelled example texts).
- Task options: `multi_label: true` (one yes/no question per label) and `ordinal: true` (distances in the
  results).
- Progress: long steps print progress, speed and time left and write `RUN/status.json`; `tinydistill status`
  reads it.
- `tune` compares teachers, few-shot settings and prompts on a dev split of the training data.
- `analyse` writes an error-analysis page; `suggest` picks texts to label by hand, and `train --gold-file`
  uses them.
- `predict`/`serve --log` record the texts the cascade sent to the teacher; `refresh` retrains with them.
- `label --shortlist K` scores only the K labels most similar to each text, for tasks with many labels;
  `eval` reports the shortlist recall.
- `--student static`: a static-embedding student (no attention), much smaller and faster.
- `report` rebuilds `results.md`; the cascade latency uses the fastest timed teacher.

## 0.2.0

- First release on PyPI.
- Teacher scoring in one forward pass per text with a prefix tree of the labels; labels are scored over their
  surface forms (lowercase, Capitalised, UPPERCASE, with and without a leading space), configurable per task.
- Label ids become natural-language names; optional label descriptions.
- `train --gold-per-class K` and `--upper-bound`; `label --cache-dir` to share teacher scores between runs.
- 4-bit teachers (`--load-4bit`).

## 0.1.0

- `label`, `train`, `export`, `eval`, `predict` and `run`; ONNX export with int8 quantisation.
