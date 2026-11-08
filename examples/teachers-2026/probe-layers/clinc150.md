| Teacher | Prompt | Probe reads | Teacher accuracy | Macro-F1 | Labelling |
|---|---|---|---|---|---|
| `google/gemma-4-12B-it` | zero-shot | LM head input (default) | 0.891 | 0.914 | 528 s |
| `google/gemma-4-12B-it` | zero-shot | layer -6 | 0.886 | 0.911 | 517 s |
| `google/gemma-4-12B-it` | zero-shot | layer -12 | 0.889 | 0.915 | 522 s |
| `google/gemma-4-12B-it` | zero-shot | layer -18 | 0.859 | 0.880 | 549 s |
| `google/gemma-4-12B-it` | zero-shot | layer -24 | 0.687 | 0.743 | 564 s |
| `google/gemma-4-12B-it` | zero-shot | layer -30 | 0.673 | 0.723 | 589 s |

Test accuracy of the probe alone (`label --probe-only`) on 5,500 test texts.
