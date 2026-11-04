| Teacher | Prompt | Few-shot examples per class | Dev accuracy | Dev macro-F1 | Seconds per text |
|---|---|---|---|---|---|
| `Qwen/Qwen3.5-9B` | task.yaml | 8 | 0.850 | 0.836 | 6.652 |
| `google/gemma-4-12B-it` | task.yaml | 8 | 0.840 | 0.831 | 11.957 |
| `Qwen/Qwen3.5-4B` | task.yaml | 8 | 0.804 | 0.795 | 6.619 |
| `google/gemma-4-E4B-it` | task.yaml | 8 | 0.794 | 0.780 | 3.917 |
| `Qwen/Qwen2.5-7B-Instruct` | task.yaml | 8 | 0.768 | 0.744 | 0.187 |
| `google/gemma-4-12B-it` | task.yaml | 0 | 0.712 | 0.680 | 2.936 |
| `Qwen/Qwen3.5-9B` | task.yaml | 0 | 0.668 | 0.641 | 1.471 |
| `Qwen/Qwen3.5-4B` | task.yaml | 0 | 0.638 | 0.611 | 1.268 |
| `Qwen/Qwen2.5-7B-Instruct` | task.yaml | 0 | 0.616 | 0.571 | 0.115 |
| `google/gemma-4-E4B-it` | task.yaml | 0 | 0.572 | 0.541 | 0.674 |

Scored on 500 gold-labelled training texts held out from the few-shot examples (not the test set), best first.

Best: `tinydistill run examples/banking77/task.yaml --teacher Qwen/Qwen3.5-9B --few-shot 8 --data ... --out ...`
