# Changelog

## 0.6.1 (2026-11-09)

- Published few-shot results, read in the papers and not rerun: on Banking77, CPFT, SetFit and LLaMA-2 7B with
  retrieved examples at 5 and 10 labels per class join the table in "How it compares", at the precision the
  papers report; on HWU64, CPFT, SetFit, DeBERTa and LLaMA models in RESULTS.md. No paper uses 8 labels per
  class, so the comparisons say where 8 sits between 5 and 10. On CLINC150 the papers score the 150 intents on
  the in-scope test texts, so `examples/clinc150/in_scope.py` scores saved runs that way: the student of the
  recommended settings reaches 0.951 deployed, against CPFT, SetFit, DeBERTa and LLaMA models in RESULTS.md.
- `examples/benchmark/check.sh`: before a release, label and train one fixed benchmark from scratch and fail
  if the deployed student lost more than a point of accuracy, or labelling took 1.5 times as long, against its
  saved reference.
- The Python API's `Distiller` takes `few_shot` and `few_shot_similar`, as `label` does, and with gold labels
  in `fit` shows each text its 16 most similar gold-labelled texts by default (when the prompt has
  `{examples}`); `few_shot_similar=0` turns that off. The examples are part of the cache key, as on the command
  line.
- `label` and `run` with `--gold-per-class K` and neither `--few-shot` nor `--few-shot-similar` show each text's
  16 most similar of those K gold examples per class in the teacher's prompt, the setting of the best runs,
  when the prompt (and every `--prompts` file) has `{examples}` before `{text}` and the task is single-label;
  `--few-shot-similar 0` shows none. The example scripts that run without examples now say
  `--few-shot-similar 0`.
- The Python API's `Distiller` fits the probe on the teacher's hidden states when `fit` gets gold labels, as
  `tinydistill label` does by default (`probe=False` turns it off), and `probe_only=True` makes the probe the
  whole teacher, so the teacher scores no label. The states are cached with the scores, in the files the
  command line uses.
- `run --dry-run` says "1 seed", not "1 seeds".
- RESULTS.md: HWU64 (64 intents), a task none of the settings were tuned on (`examples/hwu64/run.sh`). With
  the recommended settings and `--probe-only`, the deployed student reaches 0.888 after 16 minutes of
  labelling, above its teacher (0.879), against 0.639 for the student on the 512 gold labels alone. The
  teacher scoring every label as well takes 7.7 hours for a deployed student of 0.892.

## 0.6.0 (2026-11-08)

Highlights:

- Labelling in minutes: `label --probe-only` now computes only the teacher's hidden states, which is all the
  probe reads, and scores no label. On CLINC150 the deployed student reached 0.851 after 37 minutes of
  labelling, against 0.859 after 7.2 hours when the labels are scored too; with a zero-shot prompt, Banking77
  took 74 seconds.
- The probe on the teacher's hidden states is on by default whenever a run has gold labels.
- A live page that follows a run (`--live`): progress, the teacher's labels as they come in, the student's loss
  and agreement with the teacher as it trains, and the results. On a terminal, a progress bar.
- `--dry-run` says how long labelling and training will take with a local teacher.
- New Hub and live demo models: Danish MASSIVE 0.815 (from 0.759), CLINC150 0.859 (from 0.849).

Upgrade notes:

- The probe (`--probe`) is now on by default whenever a run has gold labels (`--gold-per-class`,
  `--gold-file`) and a local teacher; it lifted the deployed student on every task and teacher measured.
  `--no-probe` turns it off. It stays off with `--vllm-url`, `--api-url`, `--chunk-tokens` and `--reason`, and
  when there is no gold-labelled training text to fit on. Texts already in a cache filled without it are scored
  once more, to record their states.
- `--probe-only` refuses `--few-shot-sets`.
- Runs write `status.json` every 2 seconds (it was every 10), and also `status-log.jsonl`, `history-live.json`,
  `labels-live.json` and `command.json` for the live page.

### Teachers and labelling

- `label --probe-only` computes only each text's hidden state and scores no label. Its caches have a
  fingerprint of their own, and `--probe-dev` reports the probe's accuracy alone.
- `label --probe-layer L`: the probe reads the output of the teacher's layer L instead of the input of its LM
  head; that layer's states are cached in a file of their own.
- `label --dry-run` and `run --dry-run` with a local or vLLM teacher: how long labelling the texts not in the
  cache yet would take, timed on a sample of them after a few to warm up, and with `run` also how long training
  would take, from a few timed batches of the student. Nothing is cached or written. Before, `--dry-run` was
  for API teachers only.
- `tinydistill cache DIR` lists the score caches, largest first, with their texts, size, last write, teacher
  and the runs that use them; `--prune` deletes the ones no run uses (not those written in the last
  `--keep-recent` hours, 24 by default).

### Following a run

- `--live` (on `run`, `label`, `train` and `status RUN`) serves a page on `http://127.0.0.1` that follows the
  run every 2 seconds: the stage, time left and speed over time; the teacher's labels on the training texts so
  far, so a teacher collapsing onto a few labels shows within minutes; every few batches of training, the
  loss, the student's agreement with its target's top label and the learning rate, with held-out loss and
  agreement per epoch; the stages done; the results, with a link to `results.md`; and the command that started
  the run. The tab's title shows the progress, the page can notify you when the run's process ends, and a page
  whose server stopped says so.
- A progress bar on a terminal (`stage  done/total [=====>....]  pct  rate  ETA`, during training also the last
  epoch's loss); written to a file or a pipe, a line every 10 seconds as before. `TINYDISTILL_PROGRESS=bar` or
  `lines` chooses.
- `tinydistill status DIR` for a directory of runs prints a table of them all: stage, progress, running,
  finished or stopped, last update and the deployed student's accuracy. `status --watch` shows it again every
  2 seconds.
- `tinydistill env` shows each GPU's free and total memory.

### Students

- `train` notes when the training texts do not look English to the student's tokenizer (under 8% common English
  words and over 1.5 of its tokens per word) and suggests a multilingual student.

### Results

- The probe alone as the teacher on Banking77, CLINC150 and Danish MASSIVE
  (`examples/teachers-2026/probe-only.sh`): with 16 similar examples the deployed students are 0.3 to 1.3
  points below the runs that also score the labels, after a third to a twentieth of their labelling time.
- Which layer the probe reads (`examples/teachers-2026/probe-layers.sh`): a layer about a quarter of the way
  back was 1.5 to 1.6 points better on Banking77 and Danish MASSIVE and slightly worse on CLINC150; layers
  nearer the middle were far worse. The default stays the LM head's input.
- Danish MASSIVE: the trimmed multilingual student on the labels of Gemma 4 12B with the probe reaches 0.815
  deployed, above that teacher (0.808); on the Qwen3.5-9B and Gemma 4 12B ensemble's labels, 0.810. It is the
  new Danish Hub and live demo model, and "How it compares" uses it.
- CLINC150: the probe on Gemma 4 12B with 16 similar examples lifts the teacher from 0.914 to 0.941 and the
  deployed student from 0.849 to 0.859. It is the new CLINC150 Hub and live demo model.
- Financial tweets: Qwen3.5-9B, Gemma 4 12B and their ensemble with 16 similar examples; the ensemble's student
  (0.829 deployed) ties the zero-shot three-family ensemble's.
- New example scripts: `examples/teachers-2026/probe-only.sh`, `probe-layers.sh`, `probe-gemma.sh`,
  `examples/massive-da/ensemble-multilingual.sh`; `similar.sh` also runs financial tweets and takes task names.

### Docs

- "When to use it" has a table of which settings to use in which situation, with what they gave on the
  benchmarks.
- "Improving a run" describes the progress bar, `status` and the live page; "Choosing the teacher" fast
  labelling with `--probe-only`, the dry-run estimates and the cache command; "A few gold labels" adding a
  label later.
- Fixed: the deployed CLINC150 student was quoted as 0.860 in places; it is 0.859.

## 0.5.1 (2026-11-07)

- RESULTS.md: demonstration ensembling (`--few-shot-sets 3`) measured on Banking77 and CLINC150 with the 7B
  teacher; three sets of 16 similar examples were worse than one, for the teacher and the student.
- RESULTS.md: the linear probe on the teacher's hidden states (`--probe`) measured on the same runs; it lifted
  the deployed students from 0.855 to 0.872 on Banking77 and from 0.760 to 0.813 on CLINC150. Both are the new
  Hub models and live demo models; the README headline, the Hub table and "How it compares" use them.
- RESULTS.md: Qwen3.5-9B and Gemma 4 12B, each with 16 similar examples, and both as an ensemble
  (`examples/teachers-2026/similar.sh`). On Banking77 the ensemble is the most accurate teacher so far (0.879),
  and its student ties the probe student (0.875) at 23 times the labelling time per text.
  On CLINC150 Gemma 4 12B alone is the best teacher (0.914) and its student reaches 0.849 deployed, against
  0.813; it is the new CLINC150 Hub and live demo model, and "How it compares" and the README use it.
  On Danish MASSIVE the ensemble helps (teacher 0.806, English MiniLM-L6 student 0.771, 0.764 deployed), a
  little above the trimmed multilingual student on the Hub (0.759 deployed), which stays.
- `examples/compare/table.py`: a tie at three decimals goes to the better int8 model, then the smaller one; an
  ensemble read from the score cache is given its members' labelling time.
- The "How it works" flowchart in the README and the docs shows the probe; the README's copy matches the docs'.
- Fixed: on the "Inside the model" page, switching to a model with fewer labels left the last model's extra
  label rows in the probabilities list, where their names overlapped the new ones.
- Fixed: picking the model in use while another one downloads now calls the switch off; before, the page
  switched anyway when the download finished.

## 0.5.0 (2026-11-05)

Highlights:

- Two new teacher options from the literature: demonstration ensembling (`--few-shot-sets`) and a linear probe
  on the teacher's hidden states as one more ensemble member (`--probe`). Their benchmark runs are scripted
  (`examples/teachers-2026/`) and their results go to RESULTS.md when they finish.
- Measured in RESULTS.md: with each text's 16 most similar gold examples and "none of these" as the outside
  label, the CLINC150 student went from 0.574 to 0.767 deployed; on AG News, the fixed `tune --propose` lifted
  the 1.5B teacher's student from 0.781 to 0.804 deployed; Qwen3.5-9B as the Banking77 teacher, 0.859 deployed.
- A model menu on the browser pages, with the models loading from the Hugging Face Hub (`web --models`,
  `--remote`; `push` uploads what the pages need).
- "How it compares", a new docs page: on four tasks, the best tinydistill student against gold-only training,
  SetFit, GLiClass zero-shot and the LLM itself, with accuracy, CPU latency, size and build time.
- A SetFit-style contrastive warm-up for the student (`train --warmup-pairs`). It helps on a small labelling
  budget only, so it is off by default.

Upgrade notes:

- `run --budget N` now picks a random sample by default (`--rounds` defaults to 1, was 4); pass `--rounds 4` for
  the earlier behaviour.
- `--gold-per-class` and `--gold-file` are now also `label` options.
- Score caches stay valid: without the new options, teacher fingerprints are unchanged.

### Teachers and labelling

- Demonstration ensembling: `label --few-shot-similar M --few-shot-sets K` scores each text K times, set k
  showing it ranks k·M+1 to (k+1)·M of its most similar gold examples, and averages the label probabilities
  (Khalifa et al., 2023). Each set is a member of the teacher with its own score cache, as in a prompt ensemble.
  Local, vLLM and hosted API teachers. The default, one set, leaves fingerprints and caches unchanged.
- `label --probe`: a linear probe on the teacher's hidden state at each text's last token (prompt-augmented
  linear probing, Cho et al. 2023), fit on the gold labels train uses (`--gold-per-class`, `--gold-file`), is one
  more ensemble member. The states are recorded during the normal scoring pass and cached next to the scores.
  `--probe-dev` reports teacher accuracy on held-out gold texts (in context, probe, combined), `--probe-only`
  uses the probe alone, and `probe.pt` lets `eval` and the cascade rebuild the teacher. `--gold-per-class` and
  `--gold-file` are now also `label` options. Local teachers only.
- `run --budget N` labels a random sample of N texts by default (`--rounds` now defaults to 1, was 4): on Banking77
  the uncertainty rounds were worse than a random sample of the same size at every budget tried (1,000, 2,500 and
  5,000 of 10,003 texts). `--rounds R` still picks each later round's texts by the student's uncertainty.

### Students

- Contrastive warm-up: `train --warmup-pairs N` first trains the student's encoder on N SetFit-style pairs
  (Tunstall et al., 2022), half of the same class and half of different classes, with a cosine-similarity loss,
  then distils as usual. Classes are the gold labels and, for other texts, the teacher's top label where its
  probability is at least `--warmup-confidence` (default 0.9). `--warmup-epochs` (default 1). Single-label tasks
  only. Off by default; `meta.json` records the settings and `refresh` and `improve` reuse them.

### Browser pages and the Hub

- Several models in one site, loaded from the Hub: `web --models hf:user/a,hf:user/b` adds a "Switch model"
  menu to both pages (task, student, size and test score of each, read from its `tinydistill-page.json` at build
  time; a model without one is skipped). A pick loads in the browser from the Hub without a reload, with
  download progress; `?model=user/name` links to it, the tabs keep it, and a model that fails to load leaves the
  previous one. "Inside the model" lists only models with its view. `web hf:user/name --remote` writes the pages
  with no model files; the page's own model loads from the Hub too. `push` uploads the files this needs in the
  same commit: `onnx/model_quantized.onnx` (or `onnx/model.onnx`), the explainer's
  `onnx/explorer_quantized.onnx` and `tinydistill-page.json`, the data each page embeds (`web.site_data`,
  `explainer.explainer_data`). `Predictor("hf:...")` skips the `onnx/` copies.
- `push --title` and `--note`: the model's name and the footer line the pages show for it, saved in its
  `tinydistill-page.json`; the footer note now changes with the model picked in the menu.

### Benchmarks and documentation

- A comparison benchmark, on the new docs page "How it compares": `examples/compare/` runs the gold-only
  baseline, SetFit and GLiClass zero-shot on the CPU for Banking77, CLINC150, Danish MASSIVE and financial
  tweets, with the same 8 gold labels per class and the same test set as tinydistill, re-times the saved
  students, and `table.py` writes one table per task beside the LLM and the best saved tinydistill student
  (accuracy, CPU latency, size, build cost). Published 10-shot Banking77 results are listed separately.
  `examples/update_readme.py` also fills that page.

### Fixes

- Fix: `tune --propose` kept no rewrites from small teachers (Qwen2.5-1.5B-Instruct dropped or invented
  placeholders, so every rewrite was dropped). The teacher now rewrites only the instruction before the
  prompt's first placeholder, shown the rest with the placeholders as plain descriptions, and the rest is kept as
  written. `tune` logs why each dropped reply was dropped. A prompt that starts with a placeholder is still
  rewritten whole.
- `tune --propose` no longer drops a two-label rewrite for naming both labels; a copied option list is dropped
  only from three labels on.
- Tests read their files as UTF-8, so they pass on Windows, and skip a tokenizer that older transformers versions
  cannot convert.
- CI runs the tests offline when the Hub model cache is a full hit, so Hub rate limits no longer slow them down.

## 0.4.0 (2026-11-05)

Highlights:

- Better teachers: each text's prompt can show the gold examples most similar to it (on Banking77 the 7B
  teacher went from 0.783 with 8 examples per class to 0.851 with the 16 most similar), prompt ensembles, a
  teacher that rewrites its own instruction (`tune --propose --rounds`), a reasoning step before the answer,
  and newer models such as Qwen3.5 and Gemma 4.
- Cheaper labelling: a budget of texts chosen by the student's uncertainty, cost estimates before calling a paid
  API, and hosted APIs without log-probabilities.
- New kinds of task: long documents read in windows, an "outside" label for texts that fit none of the
  others, and much smaller multilingual students with vocabulary trimming.
- Rounds of teacher-written texts for the student's confusions (`improve`), and a review list for a person.
- "Inside the model": a browser page that shows every layer and attention head of the student as it reads a
  text, with a guided tour.

Upgrade notes:

- `web` now writes "Inside the model" as `site/index.html` and the earlier page as `site/simple/`. Running it
  over a folder from an earlier version turns `site/explainer/` into a page that forwards to the new address.
- `refresh` trains with the run's own training settings instead of the defaults.
- Score caches stay valid, except for teachers whose chat template opens a thinking block by default (Qwen3
  style): their prompt changed (see Fixes), so they score again.

### Teachers and labelling

- `label --few-shot-similar M`: each text's prompt shows the M gold training examples most similar to it
  (sentence embeddings, as for `--shortlist`; `--few-shot-model` picks the model), most similar last, never the
  text itself (Liu et al., 2021). With `--few-shot K` they are picked from those K per class. The prompt needs
  `{examples}` before `{text}`; the part before it stays cached. Local, vLLM and hosted API teachers. Score
  caches of runs without it are unchanged.
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
- `label --reason [--reason-tokens K]`: the teacher thinks briefly before it answers (greedy, up to K tokens,
  default 256), and every label is scored as the continuation of prompt + reasoning + `Answer:` with the usual
  scoring. The reasoning is cached with the scores and written to `reasoning.jsonl`; `analyse` shows it for the
  teacher's mistakes. The setting is part of the cache fingerprint and of `meta.json`. Local and vLLM teachers,
  single-label tasks; much slower than scoring alone. Without the flag nothing changes.
- Teachers that cannot pack texts (linear-attention or sliding-window layers, such as Qwen3.5 and Gemma 4) score
  many texts per forward pass instead of one: each text and each label continuation is a batch row of its own,
  texts of the same token length run together, and their continuations then run as rows against copies of each
  text's cache. Rows never see each other and need no padding before a scored token, so the scores are the same.
  A pass holds as many rows as fit in a quarter of the free GPU memory (at least 2 GiB, as before) and
  `--teacher-tokens`.
- Hosted API teachers: `label --api-url URL --teacher MODEL` scores with an OpenAI-compatible chat API
  (OpenAI, DeepInfra, Together, a llama.cpp server, ...). The labels are shown as lettered options and scored
  by the answer letter's log-probability (single-label tasks, at most 20 labels); the key comes from the
  environment variable named by `--api-key-env` and is never written to disk.
- `label --api-url URL --api-samples N [--api-temperature T]`: for hosted APIs without log-probabilities, the
  teacher samples N answers per text (default temperature 0.7) and each label's soft label is its share of the
  answers whose first word is an option letter. N and T are part of the cache fingerprint and of `meta.json`.
- `label --api-url URL --dry-run [--price-in P --price-out Q]`: prints the requests and approximate tokens
  (characters / 4) labelling the texts not in the cache would take, and with prices per million tokens the
  cost; scores nothing.
- `run --budget N [--rounds R]`: the teacher labels only N training texts, in R rounds (default 4): N/R at
  random, then in each round the N/R unlabelled texts the student trained so far is least sure of (highest
  entropy; multi-label, mean binary entropy), retraining after each round. For paid API teachers. Training
  itself is unchanged. Without `--budget` nothing changes.
- Long documents: `label --chunk-tokens T [--chunk-stride S]` cuts texts longer than T tokens of the student's
  tokenizer (`--chunk-tokenizer`) into overlapping windows, S tokens apart (default T // 2). The teacher scores
  every window and the student trains on them; a test document's teacher scores are the mean of its windows'
  probabilities. T and S are stored with the model, and `Predictor`, `predict`, `serve`, the cascade, `eval`
  and `Distiller` read every input in the same windows and average their probabilities, one row per input.
  The web page reads only the first window. Without the flag nothing changes.
- Task files take an optional `outside` key: one more label, listed last, for texts that fit none of the others
  ("none of these" by default; `id`, `name` and `description` can be set). The teacher scores it and the
  student learns it like any other label. `eval` and `results` add in-scope accuracy and out-of-scope recall
  and precision when the test labels include it; `describe()`, `predict` and `serve` mark it with
  `"outside": true`, and the web pages answer "None of the labels fits". Multi-label tasks refuse it.
  `examples/clinc150/outside.sh` runs CLINC150 with it.

### Improving a run

- `improve RUN --rounds R --per-round N`: rounds of performance-guided distillation (PGKD, Di Palo et al.,
  2024). Each round finds the label pairs the student confuses most on a held-out share of the training texts
  (`--held-out`, default 0.1, against the teacher's labels; never the test set), has the teacher write N texts
  that are easy to mistake for the other label of a pair, labels them with the run's teacher and trains,
  exports and evaluates a new student with the run's training settings in `RUN-improve/round-k`. Round 0 is
  the run retrained without the held-out texts. It stops when the held-out agreement does not improve and
  writes `RUN-improve/improve.md` and `improve.json`. Single-label runs with a local teacher.
- `refresh` trains the new student with the run's own training settings (as `improve` does) instead of the
  defaults; a training flag given with another value than its default still wins.
- `synth --vary`: each request also names a length, a tone and a writer's situation, sampled with `--seed`
  (AttrPrompt, Yu et al., 2023); a task file's `attributes` mapping replaces them. `synth --dedup T [--data
  FILE]` drops texts whose cosine similarity to a text kept earlier, or to one in FILE, is T or more. Without
  the flags the output is unchanged.
- `analyse RUN --review OUT.csv`: a CSV of training texts for a person to check, the texts whose gold label the
  teacher contradicts (its most confident first), then for ensembles the texts its members disagree on most.

### Students

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
- ModernBERT students are tested (training, ONNX and int8 export). On Banking77, ModernBERT-base reached the
  accuracy of MiniLM-L6 at about eight times the latency.

### Browser pages

- `web`: the detail panel of "Inside the model" (attention matrix, embeddings, token map, classifier) takes about
  half of the middle column instead of a fixed 290 pixels, and each drawing is drawn to fit its box, so nothing is
  cut off and the panel no longer scrolls its controls out of sight; only the text beside a drawing scrolls. A
  handle above the tabs resizes the panel (mouse, touch or arrow keys; double-click resets) and the page remembers
  the size. "Expand" opens the panel in a large view over the page.
- `web` writes two pages, and the visitor's text goes along between them (`?text=`):
  - "Inside the model", the main page (`site/index.html`): one screen with the text (rerun as you type), the
    whole network as a column of token nodes per layer with the attention lines of the chosen head (or the
    mean over heads), and the probabilities with a sparkline per label across the layers, teacher against
    student and the training loss. Hovering a token highlights it everywhere; hovering a layer shows its heads,
    clicking one pins it. Tabs under the network hold one head's attention matrix, the embeddings, a PCA map of
    the token vectors per layer and the classifier's terms, with "Math" for each one's formulas and current
    numbers (KaTeX). A guided tour ("Guided tour", or `?tour=1`) walks through it in eleven steps, with numbers
    from the current run. It runs `explorer.onnx` (`export.export_explorer`: every head's attention, the hidden
    states and the classifier's input as extra outputs) and reads at most 128 tokens; BERT-, RoBERTa-, MPNet-
    and DeBERTa-style students with the run's `student/` folder.
  - "Simple" (`site/simple/`): a card of teacher against student (model, size, time per text, test accuracy;
    the student's time measured in the browser), classification as you type, the words that support the
    answer, up to 200 bundled test texts for "Random test text" and "Check it yourself" (accuracy or macro-F1,
    agreement with the teacher and milliseconds per text on that sample), a searchable list of labels, a note
    when no label reaches 50%, download progress, and "Show the math" for each step of the inference. When the
    student alone is at least as accurate as the teacher, the cascade step says so.

  A student without the layer view (static, ModernBERT, a Hub model without its run folder) gets only the simple
  page, as `site/index.html`. `build_site(..., explainer=True)` adds the tabs.

### Python, tooling and documentation

- `Distiller`: the pipeline as a scikit-learn style estimator, `Distiller("task.yaml").fit(texts)` then
  `predict`, `predict_proba`, `describe`, `score`, `save` and `Distiller.load`, with `get_params` and
  `set_params` for scikit-learn tools (without depending on scikit-learn). With `cache_dir` it shares the
  teacher's score cache with `tinydistill label --cache-dir`.
- The package ships type information (`py.typed`) and is checked with mypy in CI.
- `tinydistill env` prints the tinydistill, Python, torch and dependency versions, the platform and the GPU,
  for bug reports. Contributing: a pull request template with a code agent policy, issue forms that ask for
  `tinydistill env` and a minimal reproduction, a check that labels likely automated pull requests for a
  maintainer to review, pre-commit hooks for ruff, Dependabot, and workflow actions pinned to commit SHAs.
- Documentation site at https://andreas-hum.github.io/tinydistill/ (MkDocs): getting started, a guide per
  topic, how it works, benchmarks, and a command-line and Python API reference generated from the code. The
  live demo moved to https://andreas-hum.github.io/tinydistill/demo/.
- The README and docs link the [Distillation Playground](https://andreas-hum.github.io/distillation-playground/), which trains small networks in the
  browser on 2D toy data and compares a student from scratch, one on the teacher's hard labels and a distilled one.

### Fixes

- Fix: a teacher whose chat template opens a thinking block by default (Qwen3 style) scored the labels inside
  that block; thinking is now turned off for the teacher's prompt and for `synth`. Other templates are
  unchanged.
- Fix: the input length a student was trained with (`--max-length`) is stored with the model, and `Predictor`,
  `eval` and the web page use it; before, they always cut inputs at 128 tokens.
- Fix: an ensemble (teachers, `--prompts` or a document's windows) gave `-Infinity` to a label no member
  scored (outside every `--shortlist`, or never sampled with `--api-samples`), and multi-label log-odds above
  about 37 became `Infinity`; neither is valid JSON. The mean is now taken in log space and stays finite.
- Fix: after an interrupted run left a cut-off line in a score cache, the next score was appended to that line
  and lost when the cache was read again. The cut-off line is now ended first.
- Fix: a prompt with no tokens before `{text}` (no chat template and no BOS token) failed with an internal
  error; it is now refused with a message.
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
