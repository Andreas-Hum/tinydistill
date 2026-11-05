| Teacher | Prompt | Few-shot examples per class | Dev accuracy | Dev macro-F1 | Seconds per text |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | propose-1.5b-4.yaml | 0 | 0.880 | 0.886 | 0.006 |
| `Qwen/Qwen2.5-1.5B-Instruct` | task.yaml | 0 | 0.850 | 0.854 | 0.005 |
| `Qwen/Qwen2.5-1.5B-Instruct` | propose-1.5b-2.yaml | 0 | 0.840 | 0.851 | 0.004 |
| `Qwen/Qwen2.5-1.5B-Instruct` | propose-1.5b-3.yaml | 0 | 0.830 | 0.832 | 0.004 |
| `Qwen/Qwen2.5-1.5B-Instruct` | propose-1.5b-1.yaml | 0 | 0.760 | 0.782 | 0.004 |
| `Qwen/Qwen2.5-1.5B-Instruct` | propose-1.5b-r2-4.yaml | 0 | 0.750 | 0.774 | 0.004 |
| `Qwen/Qwen2.5-1.5B-Instruct` | propose-1.5b-r2-3.yaml | 0 | 0.730 | 0.765 | 0.004 |
| `Qwen/Qwen2.5-1.5B-Instruct` | propose-1.5b-r2-1.yaml | 0 | 0.410 | 0.363 | 0.005 |
| `Qwen/Qwen2.5-1.5B-Instruct` | propose-1.5b-r2-2.yaml | 0 | 0.360 | 0.374 | 0.005 |
| `Qwen/Qwen2.5-1.5B-Instruct` | propose-1.5b-r2-5.yaml | 0 | 0.260 | 0.103 | 0.004 |

Scored on 100 gold-labelled training texts held out from the few-shot examples (not the test set), best first.
The teacher saw mistakes on 100 other dev texts; those are not scored.

### Instructions

The teacher's rewrites change only the instruction before the first placeholder; the rest of the prompt is kept as written.

`task.yaml` (round 1): dev accuracy 0.850, label names only

```text
Here is a news headline with the start of the article. What is it about?
Options:
{labels}

News: {text}

Reply with one option, exactly as written.
```

`propose-1.5b-1.yaml` (round 1): dev accuracy 0.760, label names only

```text
What category does this headline fall into?
Options:
{labels}

News: {text}

Reply with one option, exactly as written.
```

`propose-1.5b-2.yaml` (round 1): dev accuracy 0.840, label names only

```text
What is this news article about?
Options:
{labels}

News: {text}

Reply with one option, exactly as written.
```

`propose-1.5b-3.yaml` (round 1): dev accuracy 0.830, label names only

```text
What topic does this news headline discuss?
Options:
{labels}

News: {text}

Reply with one option, exactly as written.
```

`propose-1.5b-4.yaml` (round 1): dev accuracy 0.880, label names only

```text
What is this news headline primarily about?
Options:
{labels}

News: {text}

Reply with one option, exactly as written.
```

`propose-1.5b-r2-1.yaml` (round 2): dev accuracy 0.410, label names only

```text
In what country has the author lived for at least two years?
Options:
{labels}

News: {text}

Reply with one option, exactly as written.
```

`propose-1.5b-r2-2.yaml` (round 2): dev accuracy 0.360, label names only

```text
What is this news headline primarily about?

AP: Group Discovers John the Baptist Cave

Revised Instruction:

News: A team of archaeologists has discovered what appears to be the entrance to a well-preserved cave believed to contain remnants related to the ancient prophet John the Baptist. The find could provide valuable insights into early Christian history and religious practices.
Options:
{labels}

News: {text}

Reply with one option, exactly as written.
```

`propose-1.5b-r2-3.yaml` (round 2): dev accuracy 0.730, label names only

```text
In what way does this news headline describe?
Options:
{labels}

News: {text}

Reply with one option, exactly as written.
```

`propose-1.5b-r2-4.yaml` (round 2): dev accuracy 0.750, label names only

```text
In what way does this news headline primarily convey information?
Options:
{labels}

News: {text}

Reply with one option, exactly as written.
```

`propose-1.5b-r2-5.yaml` (round 2): dev accuracy 0.260, label names only

```text
What is this news headline primarily about?
Apple Computer Inc. will open its second Japanese retail store later this month in the western Japanese city of Osaka, it said Thursday.
Options:
{labels}

News: {text}

Reply with one option, exactly as written.
```

Best: `tinydistill run examples/ag-news/propose-1.5b-4.yaml --teacher Qwen/Qwen2.5-1.5B-Instruct --data ... --out ...`
