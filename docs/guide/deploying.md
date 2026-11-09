# Deploying

`tinydistill export RUN --int8` writes the student to `RUN/onnx`: the ONNX model (and int8 with `--int8`), the
tokenizer and the labels. Everything below works from that folder, or from a model on the Hugging Face Hub as
`hf:user/name`. The int8 model has one scale per output channel of each weight matrix: on five MiniLM students
it stayed within 0.004 of the fp32 model's accuracy, against up to 0.009 with one scale per matrix, and on the
Banking77 mpnet student it lost 0.005 instead of 0.013, at the same latency
([`examples/export/per-channel.json`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/export/per-channel.json)).

## Command line and Python

```sh
tinydistill predict RUN/onnx "some text" "another text"     # JSON lines: label and confidence
tinydistill predict RUN/onnx --data texts.csv               # a whole file (.csv, .jsonl or hf:...)
tinydistill predict RUN/onnx --data sheet.csv --text-field review --out labelled.csv
```

`--text-field` names the column that holds the texts (default `text`). With `--out FILE.csv`, `predict` writes
a CSV file instead of JSON lines: every column of the input, then `predicted_label` and `predicted_confidence`,
plus `predicted_outside` for a model with an `outside` label and `predicted_by` with `--cascade`. A
multi-label model writes `predicted_labels` (separated by `|`, as gold labels are read) and one
`predicted_scores_<label>` column per label. A CSV saved by a spreadsheet program, byte order mark included,
reads as it is.

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

`tinydistill web RUN/onnx --out site` writes two static pages plus the model in the layout
[transformers.js](https://github.com/huggingface/transformers.js) expects. The int8 model then runs in the
visitor's browser with no server at all; any static host works, GitHub Pages included. The
[live demo](https://andreas-hum.github.io/tinydistill/demo/) is such a site.

- `site/index.html`, "Inside the model": the network as it reads the visitor's text (below).
- `site/simple/index.html`, "Simple": the answer, the examples and the checks, described first.

A student without the layer view (below) gets only the simple page, as `site/index.html`. Running `web` again
over a folder written by an earlier version turns its old `site/explainer/` into a page that forwards to the new
address, so links to it keep working.

The simple page:

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

The main page, "Inside the model", is a one-screen playground of the model as it reads the
visitor's text. On the left, the text (the model reruns as you type), the bundled examples and the tokens. In the
middle, the network: a column of token nodes after the embeddings and after each layer, with lines between columns
for the attention of the chosen head (or the mean over heads), each token's strongest only, and under each layer the
chance of the final answer had the model stopped there. On the right, the probabilities with a sparkline per label
across the layers, the teacher's soft labels against the student for the bundled examples, and the training loss
with this run's cross-entropy weight. Hovering a token highlights it everywhere (its nodes, its attention lines, its
row and column in the attention matrix, its point in the token map); hovering a layer shows its heads as small
multiples, and clicking a head pins it. Tabs under the network hold the details: one head's full attention matrix,
the embeddings, a map of the token vectors per layer (PCA, computed in the page) and the terms of the classifier's
score. "Math" shows each one's formulas with the current numbers, rendered with [KaTeX](https://katex.org). The
details take about half of the middle column and each drawing is sized to fit it; drag the edge above the tabs (or
focus it and use the arrow keys) to give them more or less room, and double-click it to reset. The page remembers
the size. "Expand" opens the details in a large view over the page; Escape closes it. On a phone the panes stack,
the network scrolls inside its own box and each drawing takes its own height.

Both pages have two tabs at the top, "Inside the model" (the main page) and "Simple", and the simple page has a
link under the answer, "See inside the model on this text". The
links carry the current text in the URL (`?text=...`), and each page opens on the text it is given there.

"Guided tour" in the explainer's header (or `?tour=1` in the URL) walks through the page in eleven steps: the
text, the tokens and their ids, the embedding column, the layers and their attention lines, one token followed
through every layer, the heads of a layer with one of them pinned, the attention matrix, the token map played
from the embeddings to the last layer, the classifier's terms, the probabilities with their sparklines, and the
teacher against the student with the loss. Each step spotlights its part of the page and works the page itself
(it selects the token, pins the head, opens the tab); its numbers come from the current run. The arrow keys move
between steps and Escape ends the tour, which puts the page back as it was. On a phone the step's card sits at the
bottom of the screen.

The explainer runs `explorer.onnx`, exported from the run's PyTorch student (`RUN/student`) with every head's
attention and the hidden states as extra outputs, so it reads at most 128 tokens. It needs a BERT-, RoBERTa- or
MPNet-style student; for a static or ModernBERT student, or a model from the Hub without its run folder,
`tinydistill web` writes only the simple page and says why. The Python function is
`tinydistill.explainer.build_explainer`.

### Several models, loaded from the Hub

Models pushed with `tinydistill push` (below) can share one site, and the pages can load them from the Hub
instead of keeping copies:

```sh
tinydistill web RUN/onnx --out site --models hf:user/a,hf:user/b   # a menu with this model and two more
tinydistill web hf:user/a --out site --remote                      # no model files in site/
```

- `--models` adds a "Switch model" menu to both pages: the page's own model, then each listed model with its
  task, student, size and test score. `web` reads each repository's `tinydistill-page.json` while it builds the
  pages and skips a model without one (pushed with an earlier version), saying so.
- A visitor's pick loads in the browser: the page fetches the model's `tinydistill-page.json` and its ONNX model
  from `https://huggingface.co/<repo>/resolve/main/` with transformers.js, shows the download progress, and
  swaps the labels, examples, test texts, card and summary without a reload. The text stays.
- The URL holds the choice (`?model=user/name`), so links open on it, and the tabs carry it to the other page.
  Only models in the menu can be opened this way.
- "Inside the model" lists only the models that have its view. A link to one without it opens the simple page,
  which says so.
- When a model does not load (offline, a missing file), the page says why and keeps the model it had.
- `--remote` writes the pages with no model files: their data comes from the repository's
  `tinydistill-page.json`, and the page's own model loads from the Hub too. A rebuild then changes only the
  HTML pages instead of adding the model files to the folder again, which suits a site kept in git.
  Without `--remote`, `web` copies the model next to the pages as before.

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

In the same commit, `push` uploads what the browser pages need to run the model from the Hub (`web --models`
and `--remote`, above):

- `onnx/model_quantized.onnx` (int8) or `onnx/model.onnx`, the simple page's model in the layout transformers.js
  expects, with each layer's answer and the first token's attention when the run's PyTorch student is there;
- `onnx/explorer_quantized.onnx` or `onnx/explorer.onnx`, the explainer's model, when the student has that view;
- `tinydistill-page.json`, the data each page embeds: labels, test texts with their gold and teacher labels,
  the card, the summary, the cascade and the model's settings.

The export's own files stay at the top of the repository, so `Predictor("hf:user/name")` works as before and
does not download the browser copies. A static student has no page, so it gets only the export.

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
