# Deploying

`tinydistill export RUN --int8` writes the student to `RUN/onnx`: the ONNX model (and int8 with `--int8`), the
tokenizer and the labels. Everything below works from that folder, or from a model on the Hugging Face Hub as
`hf:user/name`.

## Command line and Python

```sh
tinydistill predict RUN/onnx "some text" "another text"     # JSON lines: label and confidence
tinydistill predict RUN/onnx --data texts.csv               # a whole file
```

```python
from tinydistill import Predictor

Predictor("RUN/onnx").describe(["some text"])
```

## HTTP server

`tinydistill serve RUN/onnx` answers `POST /predict` with `{"texts": [...]}` and `GET /health`, using only the
Python standard library. It listens on 127.0.0.1 unless you pass `--host`; put it behind your usual web server
for production.

```sh
tinydistill serve RUN/onnx --port 8000
curl -s localhost:8000/predict -d '{"texts": ["Oil prices fall as demand weakens"]}'
```

## A web page

`tinydistill web RUN/onnx --out site` writes a static page plus the model in the layout
[transformers.js](https://github.com/huggingface/transformers.js) expects. The int8 model then runs in the
visitor's browser with no server at all; any static host works, GitHub Pages included. The
[live demo](https://andreas-hum.github.io/tinydistill/demo/) is such a page.

- At the top, a card sets the teacher against the student: model, size, time per text and test accuracy
  (macro-F1 for multi-label tasks). The numbers come from the run's `results.json`; the student's time is
  measured in the visitor's browser. A number the run did not record is left out.
- The answer, the top three labels and a copy of the text with the words that support the answer highlighted
  (leave one word out) update as you type. Under a top probability of 50% the page says the text may fit
  none of the labels.
- The page holds up to 200 test texts (fewer when they are long: 150,000 characters in all), one per label
  first, with their gold labels and the teacher's ten most likely labels. Example buttons show the first few;
  "Random test text" shows the student, the teacher and the gold label for one of them, each marked correct or
  wrong. "Check it yourself" runs all of them in the browser and reports accuracy (macro-F1 for multi-label),
  agreement with the teacher and milliseconds per text. They are a sample picked to cover the labels, not
  the full test set.
- "All labels" is a searchable list of the labels.
- "How does it work?" folds out each step of the inference: the word pieces, what each layer would answer,
  where the classifier looks, all label probabilities, which words matter, the teacher's soft labels and the
  cascade. "Show the steps" plays them in order. "Show the math" adds each step's formula with the current
  text's numbers: the token ids and embedding lookup, attention for `[CLS]`, the classifier's logits, softmax
  (or a sigmoid per label), and the distillation loss as training computes it, with the KL term for a test
  text. It loads [KaTeX](https://katex.org) from jsDelivr the first time it is turned on.

The model folder is named after a hash of its contents, so browsers never mix a cached older model with a new
page. The layer-by-layer view works for BERT-, RoBERTa-, MPNet- and DeBERTa-style students; for others, such as
ModernBERT and ALBERT, the page shows the answer, the probabilities and which words matter without it.

The same command writes a second page, `site/explainer/`: a guided walk through the model as it reads the
visitor's text. An overview draws every stage from the current run (text, tokens, embeddings, each layer, the
`[CLS]` vector, the classifier, the probabilities and, for the bundled examples, the teacher's soft labels);
selecting a stage zooms into it. The steps show each token's embedding vector, every attention head of every
layer as a heat map (one head in full, with lines from a token to the tokens it attends to), what the classifier
would answer after each layer, a map of the token vectors per layer (PCA, computed in the page), the terms of the
classifier's score, the probabilities, and the training loss with this run's cross-entropy weight. Each step has
a "the math" panel with its formula and the current numbers, rendered with [KaTeX](https://katex.org).

The explainer runs `explorer.onnx`, exported from the run's PyTorch student (`RUN/student`) with every head's
attention and the hidden states as extra outputs, so it reads at most 128 tokens. It needs a BERT-, RoBERTa- or
MPNet-style student; for a static or ModernBERT student, or a model from the Hub without its run folder,
`tinydistill web` writes only the main page and says why. The Python function is
`tinydistill.explainer.build_explainer`.

## Hugging Face Hub

`tinydistill push RUN/onnx --repo user/name` uploads the model with a model card (results, usage, and links to
the base model and dataset). Anyone can then use it without the run folder:

```python
Predictor("hf:user/name")
```

```sh
tinydistill predict hf:user/name "some text"
tinydistill serve hf:user/name
```

Published examples, in a [collection](https://huggingface.co/collections/Shroomy/tinydistill-6aea5b24819dbca9cb7eeb8d):

| Model | Task | Size | Test accuracy |
|---|---|---|---|
| [`Shroomy/tinydistill-banking77-minilm`](https://huggingface.co/Shroomy/tinydistill-banking77-minilm) | Banking77, 77 intents | 23 MB | 0.814 |
| [`Shroomy/tinydistill-banking77-static`](https://huggingface.co/Shroomy/tinydistill-banking77-static) | Banking77, static embeddings | 8 MB | 0.802 |
| [`Shroomy/tinydistill-financial-tweets-minilm`](https://huggingface.co/Shroomy/tinydistill-financial-tweets-minilm) | financial tweet sentiment | 23 MB | 0.820 |

## Cascade

The student is fast but less sure on some texts. A cascade lets it answer and passes the texts it is least
confident about to the teacher:

```sh
tinydistill predict RUN/onnx --cascade RUN --threshold 0.5 "some text"
tinydistill serve RUN/onnx --cascade RUN --threshold 0.5 --log cascade.jsonl
```

Texts whose top student probability is below the threshold go to the run's teacher. `eval` shows the trade-off
on the test texts (accuracy and mean latency for 0% to 100% sent on); choose the threshold on a validation split.
When the student is already more accurate than its teacher, as on Banking77, sending texts on lowers accuracy.
`--log` records the texts the teacher answered, for [`refresh`](improving.md#learn-from-production-refresh).
