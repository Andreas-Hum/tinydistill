# Many labels

The teacher scores every surface form of every label for every text. Shared prefixes are computed once, but with
hundreds of labels it gets slow.

## Shortlist

`--shortlist K` first compares each text with the label names (and descriptions) using a small sentence-embedding
model (`--shortlist-model`, MiniLM by default), and the teacher scores only the K most similar labels; the others
get probability zero.

```sh
tinydistill run task.yaml --data train.csv --test test.csv --teacher Qwen/Qwen2.5-7B-Instruct --load-4bit \
  --shortlist 20 --out run
```

`eval` reports the **shortlist recall**: how often the gold label was among the K. A gold label outside the
shortlist cannot be learned from the teacher for that text, so if the recall is low, raise K.

## Catch-all labels

A label such as "other" or "out of scope" resembles no particular text, so it is rarely shortlisted.
`--shortlist-keep` scores such labels for every text as well (comma-separated label ids):

```sh
tinydistill run task.yaml ... --shortlist 20 --shortlist-keep oos --out run
```

## On CLINC150

CLINC150 has 150 intents plus "out of scope", with `Qwen/Qwen2.5-7B-Instruct` as the teacher:

--8<-- "examples/clinc150/shortlist.md"

The plain shortlist labels six times faster and loses little on in-scope requests, but almost never offers the
teacher "out of scope". With `--shortlist-keep oos` it is more accurate than scoring every label, at the same
speed: with fewer competing labels the teacher picks "out of scope" more often when it is right.
