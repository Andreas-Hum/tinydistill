| Teacher | Prompt | Probe reads | Teacher accuracy | Macro-F1 | Labelling |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | zero-shot | LM head input (default) | 0.795 | 0.792 | from the cache |
| `Qwen/Qwen2.5-7B-Instruct` | zero-shot | layer -4 | 0.806 | 0.803 | 60 s |
| `Qwen/Qwen2.5-7B-Instruct` | zero-shot | layer -7 | 0.811 | 0.809 | 61 s |
| `Qwen/Qwen2.5-7B-Instruct` | zero-shot | layer -10 | 0.779 | 0.777 | 60 s |
| `Qwen/Qwen2.5-7B-Instruct` | zero-shot | layer -14 | 0.492 | 0.489 | 60 s |
| `Qwen/Qwen2.5-7B-Instruct` | zero-shot | layer -18 | 0.386 | 0.385 | 59 s |

Test accuracy of the probe alone (`label --probe-only`) on 3,080 test texts.
