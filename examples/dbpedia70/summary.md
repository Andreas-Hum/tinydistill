| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dbpedia70-gemma4-12b-probe-only-zeroshot | zero-shot, 4-bit, probe only `google/gemma-4-12B-it` | 0.813 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.828 ± 0.001 | 0.823 ± 0.002 | 0.907 ± 0.001 | ONNX int8 | 10.7 | 655.8 | 23 |

All latencies on the CPU with 2 threads.
