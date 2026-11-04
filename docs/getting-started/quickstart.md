# Quickstart

This distils a sentiment classifier for financial news tweets (bearish, bullish, neutral) from
`Qwen/Qwen2.5-1.5B-Instruct` into `all-MiniLM-L6-v2`. On a GPU it takes a few minutes.

## 1. Describe the task

A task is a short YAML file: a prompt with `{labels}` (the list of options) and `{text}` (the input), and the
labels. Take the one from the examples:

```sh
curl -LO https://raw.githubusercontent.com/Andreas-Hum/tinydistill/main/examples/financial-tweets/task.yaml
```

```yaml
name: financial-tweets
text_field: text
label_field: label
prompt: |-
  Here is a tweet with financial news. Classify its sentiment for the stock, company or market it is about.
  Options:
  {labels}
  {examples}

  Tweet: {text}

  Reply with one option, exactly as written.
labels:
  bearish:
    description: negative, for example a downgrade, weak results or falling prices
  bullish:
    description: positive, for example an upgrade, strong results or rising prices
  neutral:
    description: neither, for example plain news or a mixed message
```

[The task file](../guide/task-file.md) explains every field. For your own data, `tinydistill init data.csv` writes
a starter file from the labels in it.

## 2. Distil

```sh
tinydistill run task.yaml \
  --data hf:zeroshot/twitter-financial-news-sentiment:train --limit 2000 \
  --test hf:zeroshot/twitter-financial-news-sentiment:validation --test-limit 500 \
  --teacher Qwen/Qwen2.5-1.5B-Instruct --int8 --out run
```

`run` does four steps, each also a command of its own:

| Step | Command | What it does |
|---|---|---|
| Label | `tinydistill label` | the teacher scores every label for every text; cached and resumable |
| Train | `tinydistill train` | the student learns the teacher's soft labels |
| Export | `tinydistill export` | ONNX (and int8 with `--int8`) in `run/onnx` |
| Evaluate | `tinydistill eval` | `run/results.md`: accuracy, latency and size of teacher and student |

Long steps print their progress and time left, and `tinydistill status run` shows it from another terminal.

!!! tip "No GPU?"
    For a quick look on a CPU, use a smaller teacher and fewer texts:
    `--teacher Qwen/Qwen2.5-0.5B-Instruct --limit 200 --test-limit 200`. The numbers will be weak (a 0.5B teacher
    is not good at this), but every step runs.

## 3. Look at the results

```sh
cat run/results.md
tinydistill plot run        # charts in run/plots/ (needs the plot extra)
tinydistill analyse run     # run/analysis.html: weakest labels and common mistakes
```

## 4. Use the student

```sh
tinydistill predict run/onnx "Shares jump after the company beats earnings expectations"
tinydistill serve run/onnx                  # a JSON endpoint on http://127.0.0.1:8000
tinydistill web run/onnx --out site         # a static page that runs it in the browser
```

From Python, see [From Python](python.md). To make it better, see [Improving a run](../guide/improving.md).
