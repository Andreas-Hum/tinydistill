| Teacher | Prompt | Few-shot examples per class | Dev accuracy | Dev macro-F1 | Seconds per text |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | task.yaml | 0 | 0.815 | 0.818 | 1.474 |
| `Qwen/Qwen2.5-1.5B-Instruct` | narrow-world.yaml | 0 | 0.315 | 0.246 | 1.462 |
| `Qwen/Qwen2.5-1.5B-Instruct` | with-descriptions.yaml | 0 | 0.315 | 0.243 | 1.417 |

Scored on 200 gold-labelled training texts held out from the few-shot examples (not the test set), best first; timed on a CPU.
