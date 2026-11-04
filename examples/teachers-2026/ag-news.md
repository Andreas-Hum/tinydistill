| Teacher | Prompt | Few-shot examples per class | Dev accuracy | Dev macro-F1 | Seconds per text |
|---|---|---|---|---|---|
| `Qwen/Qwen3.5-4B` | task.yaml | 0 | 0.820 | 0.840 | 0.179 |
| `Qwen/Qwen3.5-9B` | task.yaml | 0 | 0.795 | 0.814 | 0.213 |
| `google/gemma-4-E4B-it` | task.yaml | 0 | 0.765 | 0.784 | 0.182 |
| `google/gemma-4-12B-it` | task.yaml | 0 | 0.750 | 0.770 | 0.228 |
| `Qwen/Qwen2.5-7B-Instruct` | task.yaml | 0 | 0.640 | 0.655 | 0.015 |

Scored on 200 gold-labelled training texts held out from the few-shot examples (not the test set), best first.

Best: `tinydistill run examples/ag-news/task.yaml --teacher Qwen/Qwen3.5-4B --data ... --out ...`
