# The task file

A task file says what to classify: the prompt the teacher sees, the labels, and where the text and the gold
labels are in your data.

```yaml
name: banking77
text_field: text        # column holding the input text
label_field: label      # column holding gold labels (ids, or integers indexing `labels`)
prompt: |-
  You are classifying messages that customers sent to their bank's support chat.
  Choose the single intent from this list that best matches the message:
  {labels}
  {examples}

  Message: {text}

  Reply with the intent exactly as written in the list.
surface_forms: [lowercase, capitalised]   # optional, see below
labels:
  - activate_my_card    # a bare id: its name is derived, "activate my card"
  - card_arrival
  # ...
```

`tinydistill init data.csv --out task.yaml` writes a starter file from the labels in your data; edit the prompt
and the names afterwards.

## The prompt

- `{labels}` (required) becomes a bulleted list of the label names.
- `{text}` (required) is the input.
- `{examples}` (optional, on its own line) holds few-shot examples; it disappears when there are none. See
  [A few gold labels](gold-labels.md).

The prompt is sent as the user message of the model's chat template; the teacher never generates text, it only
scores how likely each label is as the answer ([How it works](../how-it-works.md)).

## Labels

The **id** is what your data and the exported model use; the **name** is what the teacher scores. Each label can
be:

```yaml
labels:
  card_arrival:                       # a bare id: the name is derived, "card arrival"
  lost_card: lost or stolen card      # id: name
  bearish:                            # with a description, shown in the list but not scored
    description: negative for the stock, company or market
  Sci/Tech:                           # name and description
    name: science and technology
    description: research, computers, the internet and gadgets
```

When your gold labels are strings, the ids must match them; when they are integers (as in many Hugging Face
datasets), they index the list of labels in order.

!!! warning "Check the task file with the teacher you will use"
    Small details can change the teacher a lot. On AG News, label descriptions made `Qwen2.5-1.5B-Instruct`
    answer "world" for nearly every article (accuracy 0.305 on 200 training texts, against 0.805 without
    them), while the 7B teacher did best with narrower descriptions. `tinydistill tune` compares versions of the
    task file on a few hundred gold-labelled training texts before anything is labelled
    ([Improving a run](improving.md#check-the-setup-first-tune)).

Keep names short and close to how the model would answer: label scores are sums of log-probabilities over
tokens, so much longer names score lower.

## Surface forms

Models do not always write a label the way it is listed, for example "Bullish" for `bullish`. Each label is
scored over its common surface forms: the name as written, lowercase, Capitalised and UPPERCASE, each with and
without a leading space. `surface_forms` chooses which (`[lowercase, capitalised]` drops UPPERCASE, which
tokenises into many pieces; `[]` scores the name as written only).

## Texts that fit no label

Some texts fit none of the labels, such as a request to a banking assistant about the weather. `outside: true`
adds one more label for them, after the others:

```yaml
outside: true           # the label `outside`, named "none of these"
# or with your own id, name or description (all optional):
outside:
  id: oos               # the id your gold labels use for such texts
  name: none of these
  description: the text fits none of the other categories
```

It is an ordinary label. The prompt lists it last, the teacher scores its name like any other, and the student
learns it from the teacher's soft labels like any other. There is no confidence threshold. By default its id is
`outside`, its name "none of these" and its description "the text fits none of the other categories"; an empty
description leaves it out. Its id and name must differ from the other labels'.

Predictions return it like any other label. For such a model, `tinydistill predict`, `serve` and `describe()`
also say `"outside": true` or `false`, so an application can route these texts, and the web pages
answer "None of the labels fits" (the probability bars still show the label). When the test texts' gold labels
include it, `eval` and `results` add three columns: the accuracy on the other test texts (in scope), and the
recall and precision of the outside label.

With `--shortlist`, add `--shortlist-keep outside` (or your id): the name "none of these" resembles no text, so
the shortlist would rarely pick it ([Many labels](many-labels.md#catch-all-labels)).
`examples/clinc150/outside.sh` runs CLINC150 this way.

`outside` is for single-label tasks only; in a multi-label task a text with no label already fits none of them,
so a task file with both is refused.

!!! note "The student learns the teacher's idea of 'none of these'"
    The outside label is only as good as the teacher's judgement of what falls outside your labels. Where the
    teacher answers "none of these" for texts that do fit a label, or picks a label for texts that fit none,
    the student learns the same. Check the out-of-scope recall and precision on gold-labelled test texts that
    include such texts before relying on it.

## Attributes for synthetic texts

`tinydistill synth --vary` names a length, a tone and a writer's situation in each request for texts
([Improving a run](improving.md#more-varied-synthetic-texts-synth-vary)). An optional `attributes` mapping
replaces these with your own: each request then names one value of each attribute, sampled with `--seed`.

```yaml
attributes:
  channel: [email, chat message, phone call transcript]
  customer: [new customer, long-time customer, small business]
```

Only `synth` reads it; labelling ignores it.

## Multi-label tasks

When a text can have any number of labels, add `multi_label: true` and ask about one label at a time, with
`{label}` after `{text}`:

```yaml
multi_label: true
prompt: |
  Review: {text}
  Does the review talk about {label}? Answer yes or no.
labels: [price, delivery, product_quality]
```

The teacher answers every label's question for a text in the same forward pass, and a label's score is
log P(yes) − log P(no) over the surface forms of both answers (`answers: [yes, no]` changes them). The student
gets a multi-label head trained on the teacher's per-label probabilities, and the results show exact-match
accuracy, macro-F1 and micro-F1 with 0.5 as the cut-off. Gold labels can be lists or strings like `a|b`.
Multi-label students need more epochs; see [Training the student](training.md).

## Ordered labels

For labels with an order, such as 1 to 5 stars, add `ordinal: true`: the results then also show the mean
absolute error in label steps and the share of predictions at most one step off.

## Data

`--data` and `--test` take a `.csv` file, a `.jsonl` file, or a Hugging Face dataset as
`hf:<dataset>[:<config>]:<split>` (needs the `hf` extra). `--limit N` keeps the first N texts; some datasets
are sorted by label, so the first texts may cover only a few labels.
