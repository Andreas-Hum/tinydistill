# tinydistill

**Most of a large model's quality on one task, many times faster, on a CPU.**

`tinydistill` turns an LLM classification prompt into a small encoder classifier by knowledge distillation.
You describe the task in a short YAML file (labels, how to say each label, an instruction prompt) and give it
unlabelled texts. An instruction-tuned LLM labels them with soft targets, a small encoder learns to imitate
those targets, and the result is exported to ONNX (optionally int8) for fast CPU inference.

It continues my bachelor's project on inference optimisation by distilling LLMs.

## Results

Two tasks, both with `Qwen/Qwen2.5-1.5B-Instruct` as the zero-shot teacher and
`sentence-transformers/all-MiniLM-L6-v2` as the student. Quality is measured on gold-labelled test texts that
neither model trains on. Every table is written by the task's `run.sh` from its saved `results*.json` file in
[`examples/`](examples); no number is typed by hand.

### Banking77: the hard case

Intent detection with 77 fine-grained classes ([Banking77](https://huggingface.co/datasets/PolyAI/banking77),
CC BY 4.0), reproduced by [`examples/banking77/run.sh`](examples/banking77/run.sh).

Teacher only, no gold labels:

<!-- results: banking77/results.md -->
| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot | cpu | 0.406 | 0.367 | n/a | 13490.4 | n/a | 0.1 | 3087 | 1.0x |
| teacher, zero-shot | cuda | 0.406 | 0.367 | n/a | 264.9 | n/a | 3.8 | 3087 | 50.9x |
| student, PyTorch | cpu | 0.386 | 0.315 | 0.770 | 6.3 | 106.4 | 300.7 | 91 | 2127.4x |
| student, ONNX fp32 | cpu | 0.386 | 0.315 | 0.770 | 3.5 | 120.4 | 265.9 | 91 | 3909.0x |
| student, ONNX int8 | cpu | 0.386 | 0.316 | 0.763 | 2.2 | 49.1 | 651.2 | 23 | 6068.4x |

Task `banking77`, 10003 unlabelled training texts, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`.
CPU rows use 2 threads. Latency is the median over repeated runs. The teacher scores one text at a time, so it is timed at batch 1 and its throughput is 1 / latency. Speed-up is relative to the teacher, zero-shot on cpu.
<!-- /results -->

With 8 gold labels per class, and the student trained on all gold labels as an upper bound:

<!-- results: banking77/results-k8.md -->
Not run yet.
<!-- /results -->

### Financial tweets: an easier task

Sentiment of financial news tweets, 3 classes
([Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment),
MIT licence per the dataset card), reproduced by
[`examples/financial-tweets/run.sh`](examples/financial-tweets/run.sh). The validation split is the test set.

<!-- results: financial-tweets/results.md -->
Not run yet.
<!-- /results -->

## Quickstart

```sh
pip install "tinydistill[hf] @ git+https://github.com/Andreas-Hum/tinydistill"
curl -LO https://raw.githubusercontent.com/Andreas-Hum/tinydistill/main/examples/banking77/task.yaml
tinydistill run task.yaml --data hf:PolyAI/banking77:train --test hf:PolyAI/banking77:test \
  --out runs/banking77 --int8
tinydistill predict runs/banking77/onnx "I still haven't received my new card"
```

Labelling the full dataset with the default teacher wants a GPU. For a quick look on a CPU, add
`--teacher Qwen/Qwen2.5-0.5B-Instruct --limit 200 --test-limit 200`. `--limit` takes the first rows, and the
Banking77 splits are sorted by intent, so a limited run only sees a few intents.

`run` does every step. Each step also has its own command, and they all read and write the run directory:

| Command | What it does |
|---|---|
| `tinydistill label TASK.yaml --data D [--test T] --out RUN` | teacher scores every label of every text; cached, resumable |
| `tinydistill train RUN` | distils the student from the teacher's soft targets |
| `tinydistill export RUN [--int8]` | writes `RUN/onnx` (ONNX, tokenizer, labels) |
| `tinydistill eval RUN` | writes `RUN/results.md` and `RUN/results.json` |
| `tinydistill predict RUN/onnx TEXT... [--data FILE]` | classifies texts, prints JSON lines |

Data can be `.csv`, `.jsonl` or `hf:<dataset>[:<config>]:<split>` (needs the `hf` extra). Useful options:
`--teacher` (any Hugging Face causal LM), `--load-4bit` (bitsandbytes; install the `4bit` extra),
`--student` (any Hugging Face encoder), `--epochs`, `--lr`, `--device`, `--threads`, and `--cache-dir` (runs that
share it label each text only once). See `tinydistill <command> --help`.

**Few labels.** With `--gold-per-class K`, the student also sees the gold labels of K randomly chosen training
examples per class, through a cross-entropy term (weight `--ce-weight`, default 1) added to the distillation
loss. `--upper-bound` additionally trains the same student on all gold training labels without the teacher,
and the results table gets an upper-bound row: how far the distilled student is from fully supervised training.

From Python:

```python
from tinydistill import Predictor

clf = Predictor("runs/banking77/onnx")  # uses model.int8.onnx when present
clf.predict(["How do I top up my card?"])  # list of label names
clf.predict_proba(["How do I top up my card?"])  # numpy array, one row per text
```

### The task file

```yaml
name: banking77
text_field: text        # column holding the input text
label_field: label      # column holding gold labels (ids, or integers indexing `labels`)
prompt: |-
  You are classifying messages that customers sent to their bank's support chat.
  Choose the single intent from this list that best matches the message:
  {labels}

  Message: {text}

  Reply with the intent exactly as written in the list.
labels:
  - activate_my_card    # a bare id: its name is derived, "activate my card"
  - card_arrival
  # ...
```

`{labels}` (required) becomes a bulleted list of the label names, and `{text}` is the input. The name is what
the teacher scores. Each label can be a bare id as above, `id: name`, or carry a description that is shown in
the list (`- bullish: positive for the stock`) but not scored:

```yaml
labels:
  bearish:
    description: negative for the stock, company or market
  lost_card: lost or stolen card
```

## How the label scoring works

The teacher never generates free text. For an input x, the prompt is rendered with the model's chat template
(user message = the instruction, generation prompt for the assistant). Each label's continuation is its
verbalisation followed by the end-of-turn token, `c_1 ... c_m`, and its score is the summed log-probability

    score(label) = log p(c_1 ... c_m | prompt(x)) = sum_j log p(c_j | prompt(x), c_1 ... c_(j-1))

Multi-token labels are handled exactly, and the end-of-turn token keeps a label that is a prefix of another
(`card` versus `card arrival`) from winning by stopping early. A softmax over the label scores gives the soft
targets. The scores are cached as JSON lines keyed by a hash of the text, in a file named after a fingerprint
of the teacher, prompt and labels, so an interrupted run resumes and a changed prompt never reuses old scores.

To keep this fast, the part of the prompt before the text is run once and its key/value cache is kept. Each
text then needs a single forward pass that covers the text and all labels together: the labels' tokens form a
prefix tree (labels that start with the same tokens, such as `card arrival` and `card linking`, share nodes),
and a 4D attention mask lets each tree node attend only to the prompt, the text and its own ancestors. There is
no padding and no per-label copy of the cache. A test checks the scores against a plain forward pass per label.

The student minimises KL(teacher || student) over the training texts, plus `--ce-weight` times the
cross-entropy on the K gold labels per class chosen by `--gold-per-class`, if any (standard knowledge
distillation, Hinton et al., 2015).

## Limitations

- **The teacher caps the student.** The student learns to imitate the teacher, mistakes included. A small LLM
  choosing zero-shot among many fine-grained labels can be weak: Qwen2.5-0.5B-Instruct, for example, often
  confuses related Banking77 intents. A bigger teacher (try `--load-4bit` with a 7B model) or a better prompt
  usually helps more than anything on the student side.
- **Scores are sums, not averages.** Longer verbalisations get lower scores, so very uneven label lengths bias the
  teacher. Keep verbalisations short and of similar length.
- **The exact wording of a label counts.** The teacher scores the label name as written, with no calibration.
  Write names the way the teacher would answer: Qwen2.5 answers sentiment with "Bullish", so the lowercase name
  "bullish" loses probability to capitalised forms the scoring does not count. The financial-tweets task uses
  capitalised names for this reason.
- **Prompt and label order matter.** Zero-shot LLMs are sensitive to wording and to the order of the options.
- **Teacher cost grows with labels.** Every text is scored against every label (shared label prefixes are
  computed once), one text at a time.
- **Single-label classification only.** Inputs are truncated to 128 tokens by default (`--max-length`).
- **Timings depend on the machine.** The results table states the threads and devices used; rerun `eval` on your
  own hardware.
- ONNX export uses PyTorch's TorchScript-based exporter.

## Development

```sh
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest
```

The tests run on a CPU in a few seconds with tiny random models from `hf-internal-testing`. They check the label
scores against a direct computation, the distillation loss and metrics against hand-computed values, cache
resumption, that the ONNX export reproduces the PyTorch logits, and the whole CLI end to end.

## References

- G. Hinton, O. Vinyals, J. Dean. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). arXiv:1503.02531, 2015.
- T. Schick, H. Schütze. [Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference](https://arxiv.org/abs/2001.07676). EACL, 2021. (Verbalisers.)
- I. Casanueva, T. Temčinas, D. Gerz, M. Henderson, I. Vulić. [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807). NLP4ConvAI Workshop, 2020. (Banking77.)
- W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, M. Zhou. [MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957). NeurIPS, 2020.
- N. Reimers, I. Gurevych. [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084). EMNLP, 2019.
- Qwen Team. [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115). arXiv:2412.15115, 2024.

## Licence

[MIT](LICENSE). The Banking77 dataset is CC BY 4.0 (PolyAI); models are downloaded from the Hugging Face Hub under
their own licences.
