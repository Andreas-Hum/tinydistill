# No unlabelled data

`tinydistill synth` asks the teacher to write example inputs for each label, from the task's own instruction:

```sh
tinydistill synth task.yaml --per-label 500 --teacher Qwen/Qwen2.5-7B-Instruct --load-4bit --out texts.jsonl
tinydistill run task.yaml --data texts.jsonl --test test.csv --out run
```

This is the only place where the teacher generates text, and only inputs: the file has no label field, and the
texts are labelled by the usual scoring when you pass them as `--data`.

Generated texts are much less varied than real ones. On financial tweets, a student trained on about 1,500
generated texts reached 0.596 accuracy, against 0.825 with real tweets
([Benchmarks](../results.md#financial-tweets-an-easier-task)). Use `synth` to start without data, and replace the
texts with real ones as soon as you have some.
