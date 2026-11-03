# Changelog

## 0.3.3 (2026-11-03)

- Charts: `tinydistill plot RUN` and `tinydistill.plots` (soft labels, accuracy and speed, training curves,
  per-label F1, confusions, cascade), with the `plot` extra (matplotlib).
- Training writes `history.json` (the loss per epoch, and the held-out loss with `--val-share`).
- Results report top-3 accuracy for tasks with more than five labels.
- Two more notebooks: your own task, and deploying; the quickstart shows the charts.
- The expected "newly initialized classifier" report from transformers is no longer printed.
- Fix: the int8 model's scores for a text depended on the other texts in the same batch (dynamic
  quantisation scales activations over the batch). `Predictor` now runs it one text at a time by default;
  on a CPU that costs little throughput. Benchmarks still time real batches.
- `push` adds `base_model` and `datasets` to the model card, so the Hub links them.
- The Banking77 student is on the Hub: `Predictor("hf:Shroomy/tinydistill-banking77-minilm")`.

## 0.3.2 (2026-11-03)

- Models load straight from the Hugging Face Hub: `Predictor("hf:user/name")`, and `predict`, `serve` and
  `web` take `hf:user/name`. The model card that `push` writes shows this.
- `train --val-share S` holds out a share of the training texts, keeps the epoch with the lowest loss on them
  and stops early after `--patience` epochs without improvement; no gold labels needed.
- A quickstart notebook (`notebooks/quickstart.ipynb`) that runs on Colab.
- The README follows the layout of widely used libraries (highlights, installation, quickstart, a results
  summary); the full results are in `RESULTS.md`.
- Fix: `tinydistill status` on Windows sent Ctrl+C to the process it checked; it now only queries it.
- Fix: files are read and written as UTF-8 on every platform (Windows defaulted to its legacy code page).
- `numpy` is a declared dependency, and the minimum torch is 2.6 (transformers needs it to load `.bin`
  weights); CI now tests the oldest supported versions.

## 0.3.1 (2026-11-03)

- Students train for 10 epochs by default (was 5). On Banking77, MiniLM-L6 goes from 0.729 to 0.808 test
  accuracy; on financial tweets, longer training changes nothing. Pass `--epochs 5` for the old behaviour.
- `label` does not load the teacher when every text is already in the cache, so training new students on
  cached labels needs neither the time nor the GPU for the teacher.
- `web`: an "Inside the model" view: tokens, the answer after each layer, attention, all label
  probabilities, which words matter, the teacher's soft labels for the examples, the cascade, and a slow-motion
  mode. The model folder is named after its contents, so browsers never reuse an older cached copy.

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
