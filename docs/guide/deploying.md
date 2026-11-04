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
visitor's browser with no server at all; any static host works, GitHub Pages included. Below the answer, an
"Inside the model" view shows each step of the inference: the word pieces, what each layer would answer, where
the classifier looks, all label probabilities, which words matter, the teacher's soft labels for the examples,
and the cascade. The [live demo](https://andreas-hum.github.io/tinydistill/demo/) is such a page.

The model folder is named after a hash of its contents, so browsers never mix a cached older model with a new
page. The layer-by-layer view works for BERT-, RoBERTa-, MPNet- and DeBERTa-style students; for others, such as
ModernBERT and ALBERT, the page shows the answer, the probabilities and which words matter without it.

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
