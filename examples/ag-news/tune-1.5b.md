| Teacher | Prompt | Few-shot examples per class | Dev accuracy | Dev macro-F1 | Seconds per text |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | task.yaml | 0 | 0.805 | 0.806 | 0.007 |
| `Qwen/Qwen2.5-1.5B-Instruct` | narrow-world.yaml | 0 | 0.315 | 0.246 | 0.005 |
| `Qwen/Qwen2.5-1.5B-Instruct` | with-descriptions.yaml | 0 | 0.305 | 0.229 | 0.005 |

Scored on 200 gold-labelled training texts held out from the few-shot examples (not the test set), best first.

Best: `tinydistill run examples/ag-news/task.yaml --teacher Qwen/Qwen2.5-1.5B-Instruct --data ... --out ...`
