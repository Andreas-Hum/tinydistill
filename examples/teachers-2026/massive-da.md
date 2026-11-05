| Teacher | Prompt | Few-shot examples per class | Dev accuracy | Dev macro-F1 | Seconds per text |
|---|---|---|---|---|---|
| `google/gemma-4-12B-it` | task.yaml | 8 | 0.814 | 0.779 | 10.780 |
| `Qwen/Qwen3.5-9B` | task.yaml | 8 | 0.798 | 0.772 | 6.159 |
| `Qwen/Qwen3.5-4B` | task.yaml | 8 | 0.780 | 0.709 | 6.020 |
| `google/gemma-4-E4B-it` | task.yaml | 8 | 0.752 | 0.740 | 3.662 |
| `google/gemma-4-12B-it` | task.yaml | 0 | 0.730 | 0.717 | 2.609 |
| `Qwen/Qwen2.5-7B-Instruct` | task.yaml | 8 | 0.700 | 0.701 | 0.180 |
| `Qwen/Qwen3.5-9B` | task.yaml | 0 | 0.696 | 0.670 | 1.459 |
| `google/gemma-4-E4B-it` | task.yaml | 0 | 0.690 | 0.680 | 0.728 |
| `Qwen/Qwen3.5-4B` | task.yaml | 0 | 0.688 | 0.665 | 1.422 |
| `Qwen/Qwen2.5-7B-Instruct` | task.yaml | 0 | 0.598 | 0.561 | 0.127 |

Scored on 500 gold-labelled training texts held out from the few-shot examples (not the test set), best first.

Best: `tinydistill run examples/massive-da/task.yaml --teacher google/gemma-4-12B-it --few-shot 8 --data ... --out ...`
