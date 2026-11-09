# Training the student

The student minimises the KL divergence from the teacher's soft labels to its own predictions over the
training texts, plus a cross-entropy term on any gold labels: standard knowledge distillation (Hinton et al.,
2015).

```sh
tinydistill train run --student sentence-transformers/all-MiniLM-L6-v2 --epochs 10 --seeds 3
```

## The student

`--student` takes any Hugging Face encoder. Bigger encoders are more accurate and slower:

| Student | Size (int8) | Latency per text (CPU) |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` (default) | 23 MB | about 1 ms |
| `sentence-transformers/all-MiniLM-L12-v2` | 34 MB | about 3 ms |
| `sentence-transformers/all-mpnet-base-v2` | 110 MB | about 7 ms |
| `answerdotai/ModernBERT-base` | 151 MB | about 10 ms |
| `jhu-clsp/ettin-encoder-17m` | 17 MB | about 0.7 ms |
| `jhu-clsp/ettin-encoder-32m` | 33 MB | about 1.5 ms |
| `jhu-clsp/mmBERT-small` (multilingual) | 142 MB | about 5 ms |
| `static` (below) | 8 MB | about 0.1 ms |
| `static:minishlab/potion-base-8M` (below) | 8 MB | about 0.05 ms |
| `static:minishlab/potion-multilingual-128M` (below, multilingual) | 128 MB | about 0.07 ms |

Ettin (English) and mmBERT (multilingual, over 1,800 languages) are ModernBERT encoders and train, export
and run as any other student. Their rows and the potion rows come from `examples/students-cpu/measure.py`
(`encoders.json`: 20 training steps, then the int8 model through `Predictor` at batch 1 on short texts, 2 CPU
threads; MiniLM-L6 measured 0.85 ms in the same run). That checks that they work; their accuracy as students
has not been measured. mmBERT is large for its speed because of its 256,000-token vocabulary.

On Banking77, MiniLM-L6 and ModernBERT trained for 20 epochs and mpnet trained for 5 reached about the same
accuracy (0.813, 0.815 and 0.814; [Benchmarks](../results.md#banking77-the-hard-case)), so the smallest is the
one to deploy there.

### Texts in other languages

The task file can stay in English while the texts are not: the teacher reads an English instruction over
Danish texts well enough to teach a student. For the student, pick a multilingual encoder, such as
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, mmBERT or the multilingual potion static model.
On Danish MASSIVE (60 intents), the multilingual MiniLM reached 0.761 against 0.724 for the English MiniLM-L6,
from the same 8-shot 7B teacher (0.736; [Benchmarks](../results.md#danish-massive-an-english-prompt-over-danish-texts)).
With a better teacher the gap grows: on the labels of Gemma 4 12B shown each text's 16 most similar gold
examples and averaged with a probe on its hidden states (0.808), the trimmed multilingual MiniLM reaches 0.815
against 0.768 for the English one
([Benchmarks](../results.md#the-newer-teachers-with-similar-examples-and-both-together)).
When the training texts do not look English to the student's tokenizer (under 8% common English words and
over 1.5 of its tokens per word), `train` says so and suggests a multilingual student.

#### Trimming the vocabulary

A multilingual encoder is large mostly because of its embedding matrix, one row per token of a vocabulary that
covers about a hundred languages (250,000 tokens for the multilingual MiniLM, 256,000 for mmBERT). A task in one
or two languages uses a small part of it. `--trim-vocab` keeps only the rows of the tokens the student's
tokenizer produces on the run's training texts, plus the special tokens and the `--trim-keep K`
(default 1000) most frequent tokens of the original vocabulary, before training starts (vocabulary trimming,
[Ushio et al., 2023](https://arxiv.org/abs/2305.15020)):

```sh
tinydistill train run --student sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --trim-vocab
```

On Danish MASSIVE this cut the int8 student from 118 MB to 24 MB with the same test accuracy (0.761) and
latency ([Benchmarks](../results.md#danish-massive-an-english-prompt-over-danish-texts)).

The tokenizer is rebuilt to match: its `tokenizer.json` (and `vocab.txt`, where one is written) holds only the
kept tokens, in their old order, at ids that index the kept rows. The student then trains, and is saved,
exported and served, as a model with that smaller vocabulary; `Predictor`, `serve` and `web` need nothing new.
Trimming before training rather than at export means the model that trains, the one `eval` measures and the one
that is deployed are the same, and training holds the optimizer's state for the small matrix only.
It works for WordPiece (BERT, MiniLM), Unigram (XLM-R and the multilingual MiniLM) and BPE tokenizers (mmBERT's
Gemma tokenizer, byte-level BPE such as ModernBERT's), and for the static student. `meta.json` records
`trim_vocab`, `trim_keep` and the student's `vocab_size`.

A text whose tokens were all kept gets the same tokens as with the full vocabulary, and so the same logits from
the same weights. This holds for every training text. A new text, test texts included, can contain a word whose
tokens were
trimmed. The tokenizer then splits that word into smaller kept pieces where it can. Otherwise it writes the
word as bytes (BPE tokenizers with byte fallback, such as mmBERT's) or as the unknown token (WordPiece and
Unigram). Such a text runs, but the student reads it differently from the untrimmed student, and it has never
seen those tokens in training. The most frequent tokens kept by `--trim-keep` are a margin for such texts:
for Unigram they are the pieces with the highest scores, and for WordPiece and BPE the lowest ids, which follow
frequency or merge order. The test texts are not used to choose the tokens, so the test accuracy includes
this effect, as in use.

The default of 1000 costs at most 1000 rows, under 0.4 MB at int8 for a 384-dimensional encoder such as the
multilingual MiniLM, and covers the commonest pieces of the whole vocabulary. `--trim-keep 0` keeps
only the texts' own tokens. How much accuracy a trimmed student keeps after training has not been measured yet.

## Starting from the label names

`--init-head names` sets the classifier's starting weights from the label names instead of at random: each
label's row is the student's own representation of its name (the verbalisation, plus the description if the
task has one), as the classifier receives it, normalised and scaled so that a text scores each label about 10
times its cosine similarity to the name; the bias starts at zero. Before any training the student already
ranks labels by how close a text is to their names, and training then proceeds as usual. It works for BERT,
RoBERTa/MPNet and ModernBERT students and the static student, and is off by default. It changes only the
starting point, not the loss.

```sh
tinydistill train run --init-head names
```

The representation is whatever the classifier reads: the first token through the pooler for BERT-style
students, the first token through a dense layer for RoBERTa and MPNet, the mean of all tokens through a dense
layer and norm for ModernBERT, Ettin and mmBERT, and the mean token vector for the static student. For a
multi-label task the start ranks the labels but does not yet decide which apply.

## Contrastive warm-up

`--warmup-pairs N` trains the student's encoder contrastively before distillation, as SetFit does
([Tunstall et al., 2022](https://arxiv.org/abs/2209.11055)). It draws N pairs of training texts: half are two
texts of the same class, with target 1, and half are texts of two different classes, with target 0. The loss is
the squared difference between each pair's cosine similarity and its target. The representation is the one the
classifier reads, as for `--init-head` above. The encoder trains on the pairs for `--warmup-epochs E`
(default 1), at the run's `--lr` and `--batch-size` (in pairs). Distillation then starts from the warmed-up
encoder, with the usual loss.

```sh
tinydistill train run --gold-per-class 8 --warmup-pairs 20000
```

A text's class is its gold label where it has one (`--gold-per-class`, `--gold-file`). Otherwise it is the
teacher's top label, if the teacher gives that label a probability of at least `--warmup-confidence C`
(default 0.9). The probabilities are the teacher's own, as stored in `train.jsonl`. Texts held out by
`--val-share` are not used. The pairs are drawn with the run's seed. The option is off by default, changes only
the encoder's starting point, and applies to the distilled student only, not to the `--gold-only` and
`--upper-bound` baselines. It works for transformer students and the static student. It needs a single-label
task: a multi-label task is refused. `meta.json` records `warmup_pairs`, `warmup_epochs` and
`warmup_confidence`, and `refresh` and `improve` train with them.

Measured on Banking77 ([RESULTS](https://github.com/Andreas-Hum/tinydistill/blob/main/RESULTS.md#a-contrastive-warm-up-of-the-student)),
the warm-up helps when the teacher labels only a small share of the texts (`run --budget`) and adds nothing
once it labels them all, so it is off by default.

## Long texts

Inputs are cut to `--max-length` tokens (default 128), which suits short messages. Past that, the student never
sees the rest of a text. There are two ways to handle longer texts.

**Read more tokens.** Raise `--max-length`, and consider ModernBERT, which reads up to 8,192 tokens; BERT-style
students read at most 512. The length is stored with the model, so `predict`, `serve` and the web page cut
inputs the same way. Attention costs grow with the square of the length.

```sh
tinydistill train run --student answerdotai/ModernBERT-base --max-length 1024
```

**Read in windows.** `label --chunk-tokens T` cuts every text longer than T tokens into overlapping windows of T
tokens, `--chunk-stride S` apart (default T // 2, so each token is in about two windows; S = T for no overlap).
The windows are decoded back to text and the teacher scores each one as an ordinary text. The tokens are the
student's: `--chunk-tokenizer` names the student (default: the default student; `run` uses its `--student`),
and `train` warns if its `--student` is a different one.

```sh
tinydistill label task.yaml --data docs.jsonl --test test.jsonl --out run --chunk-tokens 256 \
  --chunk-tokenizer answerdotai/ModernBERT-base
tinydistill train run --student answerdotai/ModernBERT-base
```

- `train.jsonl` holds one row per window, with the document's gold label and `doc`, the document's row number
  in the data. The student trains on the windows.
- `test.jsonl` holds one row per document. Its teacher scores are the log of the mean of its windows'
  probabilities (multi-label: of each label's probability), as an ensemble of teachers is combined.
- `train` stores T and S with the model (`tinydistill_chunk_tokens`, `tinydistill_chunk_stride` in
  `config.json`) and raises `--max-length` to T plus the special tokens if it is shorter.
- `Predictor`, and through it `predict`, `serve`, the cascade and `Distiller`, cut each input into the same
  windows, run them, and return one row per input: the mean of the windows' probabilities. In a cascade the
  teacher scores the windows too. `eval` therefore reports document-level quality.
- A text of at most T tokens is a single window and is scored exactly as without the flag.
- The web page reads only the first window of a text; `web` warns about it.

On 2,000 IMDB reviews, MiniLM-L6 read in windows of 128 tokens, 64 apart, reached 0.860 against 0.844 when
reading the first 128 tokens, at about three times the latency
([Benchmarks](../results.md#imdb-long-reviews)).

A window that starts inside a word can come to a token or two more than T when it is read again; they are cut
at `--max-length`, the same way in training and prediction. `--gold-file` matches whole texts, so it does not
reach the windows of a long document.

## Another student on the same labels

`train RUN --out NEW` trains in a new run directory with a copy of RUN's task, labels and settings, so RUN
keeps its student, and the teacher is not run again. Export and evaluate the new run as usual:

```sh
tinydistill train run --student sentence-transformers/all-MiniLM-L12-v2 --out run-l12
tinydistill export run-l12 --int8 && tinydistill eval run-l12
```

## How long to train

`--epochs` is the number of passes over the training texts (default 10). Training longer mattered more than a
bigger student on Banking77: MiniLM-L6 reached 0.729 after 5 epochs and 0.813 after 20, as accurate as mpnet at
a sixth of its latency. On the easier financial-tweets task, 20 epochs gave the same accuracy as 5.

To choose the epoch from the data, `--val-share 0.05` holds out 5% of the training texts (texts without gold
labels, so none are lost), measures the loss on them after every epoch, keeps the best epoch and stops after
`--patience` epochs (default 3) without improvement. No gold labels are needed: the held-out loss is measured
against the teacher's soft labels. Each run writes the loss per epoch to `student/history.json`, and
`tinydistill plot` draws it.

```sh
tinydistill train run --epochs 30 --val-share 0.05
```

!!! tip "Multi-label students need more epochs"
    A multi-label student has a yes/no decision per label to learn. On GoEmotions (28 labels), every seed kept
    its last epoch at 10 epochs and the student reached macro-F1 0.066; with `--epochs 30 --val-share 0.05` it
    reached 0.226, close to its teacher's 0.277.

## Seeds

`--seeds N` trains every student N times with different seeds; the results show the mean and standard deviation,
and the exported model is the first seed's.

## Static student

`--student static` (or `static:<encoder>`) trains a student with no attention at all: one vector per token,
averaged, then a linear layer, started from the encoder's own token representations (as in Model2Vec). It is
much smaller and faster than a transformer student. It ignores word order, so it can miss meaning that depends on
it (negation, "A rather than B"). On Banking77 it reached 0.803, about a point behind MiniLM students trained as
long. It runs with `predict` and `serve` like any other student, but not in the browser page. Train it longer and
at a higher learning rate:

```sh
tinydistill train run --student static --epochs 20 --lr 1e-3
```

`static:<encoder>` also takes a [Model2Vec](https://github.com/MinishLab/model2vec) model, such as
`minishlab/potion-base-8M` (English) or `minishlab/potion-multilingual-128M`. Its token vectors are read from
the model's files and used as they are (no `model2vec` package needed), so the student starts from vectors
trained for sentence embeddings rather than from the encoder's single-token outputs. The student averages every
token the tokenizer returns, including `[CLS]` and `[SEP]`, where Model2Vec drops them; training adjusts for it.

```sh
tinydistill train run --student static:minishlab/potion-base-8M --epochs 20 --lr 1e-3
```
