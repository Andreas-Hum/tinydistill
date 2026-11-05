# Changelog

## Unreleased

- `web`: "Inside the model" (the explainer) is the main page, `site/index.html`, and the simpler page moves to
  `site/simple/`; the tabs read "Inside the model" and "Simple". A student without the layer view gets only the
  simple page. An old `site/explainer/` forwards to the new address.
- Task files take an optional `outside` key: one more label, listed last, for texts that fit none of the others
  ("none of these" by default; `id`, `name` and `description` can be set). The teacher scores it and the
  student learns it like any other label. `eval` and `results` add in-scope accuracy and out-of-scope recall
  and precision when the test labels include it; `describe()`, `predict` and `serve` mark it with
  `"outside": true`, and the web pages answer "None of the labels fits". Multi-label tasks refuse it.
  `examples/clinc150/outside.sh` runs CLINC150 with it.
- Teachers that cannot pack texts (linear-attention or sliding-window layers, such as Qwen3.5 and Gemma 4) score
  many texts per forward pass instead of one: each text and each label continuation is a batch row of its own,
  texts of the same token length run together, and their continuations then run as rows against copies of each
  text's cache. Rows never see each other and need no padding before a scored token, so the scores are the same.
  A pass holds as many rows as fit in a quarter of the free GPU memory (at least 2 GiB, as before) and
  `--teacher-tokens`.
- `train --trim-vocab` cuts the student's vocabulary to the tokens of the run's training texts before
  training (vocabulary trimming, Ushio et al., 2023), plus the special tokens and the `--trim-keep K` (default
  1000) most frequent tokens of the original vocabulary. The embedding matrix keeps only those rows and the
  tokenizer (`tokenizer.json`, and `vocab.txt` where one is written) is rebuilt to match, so the exported ONNX
  model, `Predictor`, `serve` and `web` use it unchanged. It works for WordPiece (BERT, MiniLM), Unigram (XLM-R,
  the multilingual MiniLM) and BPE (mmBERT's Gemma tokenizer, byte-level BPE) tokenizers and for the static
  student. A text whose tokens were all kept gets the same tokens and logits as with the full vocabulary; other
  texts fall back to kept pieces, bytes or the unknown token. `meta.json` records `trim_vocab`, `trim_keep` and
  the student's `vocab_size`; `refresh` and `improve` keep the setting. In Python: `student.trim_vocab` and
  `train_student(trim_texts=...)`.
- `web`: the main page and the explainer have tabs to each other ("Try it", "Inside the model"), and the main
  page links to the explainer under the answer; the links carry the current text (`?text=`). The tabs appear only
  when the explainer was written. The explainer has a guided tour (its "Guided tour" button, or `?tour=1`): eleven
  steps that spotlight each part of the page and work it, with numbers from the current run. `build_site` takes
  `explainer=True` for the tabs, in place of the link `web` used to add to the footer.
- The README and docs link the [Distillation Playground](https://andreas-hum.github.io/distillation-playground/), which trains small networks in the
  browser on 2D toy data and compares a student from scratch, one on the teacher's hard labels and a distilled one.
- `refresh` trains the new student with the run's own training settings (as `improve` does) instead of the
  defaults; a training flag given with another value than its default still wins.
- `web`: the page leads with a card of teacher against student (model, size, time per text, test accuracy;
  from `results.json`, with the student's time measured in the browser), classifies as you type (no Classify
  button), and highlights the words that support the answer under it. It holds up to 200 test texts with
  their gold and teacher labels, for "Random test text" and "Check it yourself" (accuracy or macro-F1,
  agreement with the teacher and milliseconds per text on that sample, in the browser). Example buttons show
  their text. Also a searchable list of labels, a note when no label reaches 50%, download progress, and
  "Show the math", which adds each step's formula with the current text's numbers (KaTeX, loaded on first
  use). "Inside the model" is folded under "How does it work?". When the student alone is at least as
  accurate as the teacher, the cascade step says sending texts to the teacher does not help.
- `web`: the explainer page is now a one-screen playground: text (rerun as you type), a diagram of the whole network
  (a column of token nodes per layer, attention lines from the chosen head or the mean over heads) and the output
  (probabilities with a sparkline per label across the layers, teacher against student, the loss). Hovering a token
  highlights it everywhere; hovering a layer shows its heads, clicking one pins it. The attention matrix, embeddings,
  token map, classifier terms and the math moved into tabs under the diagram. On a phone the panes stack.
- `web` also writes `OUT/explainer/`, a guided walk through the student as it reads the visitor's text: an
  overview of every stage drawn from the current run, then tokens and embeddings, every attention head of every
  layer, what the classifier would answer after each layer, a PCA map of the token vectors per layer, the
  classifier's terms, the probabilities and the training loss, each with its formula and current numbers
  (KaTeX). It runs `explorer.onnx` (`export.export_explorer`: every head's attention, the hidden states and the
  classifier's input as extra outputs) and reads at most 128 tokens. BERT-, RoBERTa- and MPNet-style students with
  the run's `student/` folder; otherwise `web` says why it skipped the page. The main page is unchanged.
- `run --budget N [--rounds R]`: the teacher labels only N training texts, in R rounds (default 4): N/R at
  random, then in each round the N/R unlabelled texts the student trained so far is least sure of (highest
  entropy; multi-label, mean binary entropy), retraining after each round. For paid API teachers. Training
  itself is unchanged. Without `--budget` nothing changes.
- `label --api-url URL --dry-run [--price-in P --price-out Q]`: prints the requests and approximate tokens
  (characters / 4) labelling the texts not in the cache would take, and with prices per million tokens the
  cost; scores nothing.
- `analyse RUN --review OUT.csv`: a CSV of training texts for a person to check, the texts whose gold label the
  teacher contradicts (its most confident first), then for ensembles the texts its members disagree on most.
- `improve RUN --rounds R --per-round N`: rounds of performance-guided distillation (PGKD, Di Palo et al.,
  2024). Each round finds the label pairs the student confuses most on a held-out share of the training texts
  (`--held-out`, default 0.1, against the teacher's labels; never the test set), has the teacher write N texts
  that are easy to mistake for the other label of a pair, labels them with the run's teacher and trains,
  exports and evaluates a new student with the run's training settings in `RUN-improve/round-k`. Round 0 is
  the run retrained without the held-out texts. It stops when the held-out agreement does not improve and
  writes `RUN-improve/improve.md` and `improve.json`. Single-label runs with a local teacher.
- `synth --vary`: each request also names a length, a tone and a writer's situation, sampled with `--seed`
  (AttrPrompt, Yu et al., 2023); a task file's `attributes` mapping replaces them. `synth --dedup T [--data
  FILE]` drops texts whose cosine similarity to a text kept earlier, or to one in FILE, is T or more. Without
  the flags the output is unchanged.
- `train --init-head names`: before training, each label's classifier weights are set to the student's own
  representation of the label's name (and description), normalised and scaled, with zero bias, so the untrained
  student ranks labels by similarity to their names. BERT, RoBERTa/MPNet, ModernBERT and static students. Off by
  default; recorded in `meta.json`.
- Static students from Model2Vec models: `--student static:minishlab/potion-base-8M` (or any folder or Hub model
  with `model_type: model2vec`) starts from the model's own token vectors, read from its files without the
  `model2vec` package. Tokens the Model2Vec model has no vector for (such as `<s>` in potion-multilingual)
  start at zero.
- Ettin (`jhu-clsp/ettin-encoder-17m`, `-32m`) and mmBERT (`jhu-clsp/mmBERT-small`) students are checked:
  they train, export to ONNX and int8, and run in `Predictor` with no changes. Sizes and CPU latencies, with
  those of the potion static students, are in `examples/students-cpu/encoders.json` and the student table.
- `label --reason [--reason-tokens K]`: the teacher thinks briefly before it answers (greedy, up to K tokens,
  default 256), and every label is scored as the continuation of prompt + reasoning + `Answer:` with the usual
  scoring. The reasoning is cached with the scores and written to `reasoning.jsonl`; `analyse` shows it for the
  teacher's mistakes. The setting is part of the cache fingerprint and of `meta.json`. Local and vLLM teachers,
  single-label tasks; much slower than scoring alone. Without the flag nothing changes.
- `label --api-url URL --api-samples N [--api-temperature T]`: for hosted APIs without log-probabilities, the
  teacher samples N answers per text (default temperature 0.7) and each label's soft label is its share of the
  answers whose first word is an option letter. N and T are part of the cache fingerprint and of `meta.json`.
- Prompt ensembles: `label --prompts b.yaml,c.yaml` scores every text with the task's prompt and with each
  variant task file (same label ids in the same order) on one loaded teacher, and averages the label
  probabilities. With several teachers the members are every teacher with every prompt; each member has its
  own score cache. The variants are copied into the run (`prompt-1.yaml`, ...) and recorded in `meta.json`, so
  `eval` and the cascade use the same teacher. Local teachers only.
- `tune --propose N`: the teacher writes N rewrites of the task's instruction; those that keep its placeholders
  are scored on the dev split together with the task without label descriptions. Each candidate is written as
  `OUT-<name>.yaml`, the best as `OUT-best.yaml`, and the report lists every instruction with its dev score.
  `tune.json` rows now include the instruction.
- `tune --propose N --rounds R`: after the first round, the teacher is shown up to 8 dev texts the best
  instruction so far gets wrong and writes N rewrites of it to fix them, for up to R rounds; it stops at the
  first round without an improvement. Mistakes come from one half of the dev split and all candidates are
  scored on the other. The report and `tune.json` give each candidate's round; `OUT-best.yaml` is the best of
  all rounds. Without `--rounds` nothing changes.
- Long documents: `label --chunk-tokens T [--chunk-stride S]` cuts texts longer than T tokens of the student's
  tokenizer (`--chunk-tokenizer`) into overlapping windows, S tokens apart (default T // 2). The teacher scores
  every window and the student trains on them; a test document's teacher scores are the mean of its windows'
  probabilities. T and S are stored with the model, and `Predictor`, `predict`, `serve`, the cascade, `eval`
  and `Distiller` read every input in the same windows and average their probabilities, one row per input.
  The web page reads only the first window. Without the flag nothing changes.
- `label --few-shot-similar M`: each text's prompt shows the M gold training examples most similar to it
  (sentence embeddings, as for `--shortlist`; `--few-shot-model` picks the model), most similar last, never the
  text itself (Liu et al., 2021). With `--few-shot K` they are picked from those K per class. The prompt needs
  `{examples}` before `{text}`; the part before it stays cached. Local, vLLM and hosted API teachers. Score
  caches of runs without it are unchanged.
- ModernBERT students are tested (training, ONNX and int8 export). On Banking77, ModernBERT-base reached the
  accuracy of MiniLM-L6 at about eight times the latency.
- Fix: the input length a student was trained with (`--max-length`) is stored with the model, and `Predictor`,
  `eval` and the web page use it; before, they always cut inputs at 128 tokens.
- Fix: an ensemble (teachers, `--prompts` or a document's windows) gave `-Infinity` to a label no member
  scored (outside every `--shortlist`, or never sampled with `--api-samples`), and multi-label log-odds above
  about 37 became `Infinity`; neither is valid JSON. The mean is now taken in log space and stays finite.
- Fix: after an interrupted run left a cut-off line in a score cache, the next score was appended to that line
  and lost when the cache was read again. The cut-off line is now ended first.
- Fix: a prompt with no tokens before `{text}` (no chat template and no BOS token) failed with an internal
  error; it is now refused with a message.
- The package ships type information (`py.typed`) and is checked with mypy in CI.
- `Distiller`: the pipeline as a scikit-learn style estimator, `Distiller("task.yaml").fit(texts)` then
  `predict`, `predict_proba`, `describe`, `score`, `save` and `Distiller.load`, with `get_params` and
  `set_params` for scikit-learn tools (without depending on scikit-learn). With `cache_dir` it shares the
  teacher's score cache with `tinydistill label --cache-dir`.
- Hosted API teachers: `label --api-url URL --teacher MODEL` scores with an OpenAI-compatible chat API
  (OpenAI, DeepInfra, Together, a llama.cpp server, ...). The labels are shown as lettered options and scored
  by the answer letter's log-probability (single-label tasks, at most 20 labels); the key comes from the
  environment variable named by `--api-key-env` and is never written to disk.
- Documentation site at https://andreas-hum.github.io/tinydistill/ (MkDocs): getting started, a guide per
  topic, how it works, benchmarks, and a command-line and Python API reference generated from the code. The
  live demo moved to https://andreas-hum.github.io/tinydistill/demo/.
- `tinydistill env` prints the tinydistill, Python, torch and dependency versions, the platform and the GPU,
  for bug reports. Contributing: a pull request template with a code agent policy, issue forms that ask for
  `tinydistill env` and a minimal reproduction, a check that labels likely automated pull requests for a
  maintainer to review, pre-commit hooks for ruff, Dependabot, and workflow actions pinned to commit SHAs.
- Fix: labelling a run again without `--few-shot` (or without `--test`) left the earlier `examples.jsonl` (or
  `test.jsonl`), so `eval` and the cascade showed the teacher examples it had not labelled with (or `eval`
  reported on the earlier test labels). They are now removed.
- Fix: `label --dry-run` wrote the task, prompt and example files into the run directory, changing the teacher
  that `eval` and the cascade rebuild. A dry run now leaves the run as it was.
- Fix: `label --few-shot-similar` with `--chunk-tokens` showed each window its own document, with its gold
  label, as an example. The combination is refused.
- Fix: `refresh` added logged texts that are in the test set to the training texts. They are left out.
- Fix: `improve` trained without the run's `--init-head`.
- Fix: `export` without `--int8` left an earlier int8 model, which `Predictor` and `predict` use first. It is
  removed.
- Fix: `tune` stopped with an IndexError when no setting could be scored (every `--few-shot` value above 0 and
  no `{examples}` line), and recommended the wrong file when two task files had the same file name. Both are
  refused with a message.
- Fix: the web page of a DeBERTa student skipped the model's pooler, so its answers differed from `predict`; it
  now applies it. For ALBERT students, whose layer view failed, the page runs the plain exported model.
- Fix: the cascade table in `eval` reports, for each threshold, what a cascade at that threshold does. When
  student confidences tied at the threshold (often at 1.0), it counted texts as sent to the teacher that the
  cascade keeps.
- Fix: `Distiller.save` into a folder holding an earlier int8 model removes that file when the saved model has
  none; before, `Predictor` and `Distiller.load` read the old int8 model.

## 0.3.4 (2026-11-04)

- Fix: `tune --out NAME` with a dot in NAME (such as `tune-1.5b`) wrote `tune-1.md`; it now writes
  `NAME.md` and `NAME.json`.
- `label --shortlist-keep IDS`: labels the teacher scores for every text besides the shortlist, for catch-all
  labels such as "other" or "out of scope" (on CLINC150, "out of scope" was shortlisted for only 9% of its
  test texts).
- Fix: with many labels, teacher scoring used up GPU memory (a float32 log-softmax over the whole vocabulary
  for every kept row) and slowed to a crawl; it now takes only the target tokens' log-probabilities. CLINC150
  (151 labels) labels at about 30 texts per second instead of under 2, with 10 GB less memory.

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
