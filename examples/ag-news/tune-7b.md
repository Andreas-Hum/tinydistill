| Teacher | Prompt | Few-shot examples per class | Dev accuracy | Dev macro-F1 | Seconds per text |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` (4-bit) | narrow-world.yaml | 0 | 0.785 | 0.804 | 0.016 |
| `Qwen/Qwen2.5-7B-Instruct` (4-bit) | task.yaml | 0 | 0.640 | 0.655 | 0.018 |

Scored on the same 200 gold-labelled training texts as tune-1.5b.md, best first; timed on the GPU.
