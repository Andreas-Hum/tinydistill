| Teacher | Prompt | Probe reads | Teacher accuracy | Macro-F1 | Labelling |
|---|---|---|---|---|---|
| `google/gemma-4-12B-it` | zero-shot | LM head input (default) | 0.760 | 0.740 | from the cache |
| `google/gemma-4-12B-it` | zero-shot | layer -6 | 0.764 | 0.744 | 1116 s |
| `google/gemma-4-12B-it` | zero-shot | layer -12 | 0.775 | 0.752 | 874 s |
| `google/gemma-4-12B-it` | zero-shot | layer -18 | 0.722 | 0.691 | 838 s |
| `google/gemma-4-12B-it` | zero-shot | layer -24 | 0.538 | 0.493 | 841 s |
| `google/gemma-4-12B-it` | zero-shot | layer -30 | 0.537 | 0.500 | 918 s |

Test accuracy of the probe alone (`label --probe-only`) on 2,974 test texts.
