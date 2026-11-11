| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| argument-topic71-gemma4-12b-probe-only-zeroshot | zero-shot, 4-bit, probe only `google/gemma-4-12B-it` | 0.977 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.960 ± 0.000 | 0.958 ± 0.000 | 0.957 ± 0.001 | ONNX int8 | 2.4 | 69.1 | 23 |
| argument-topic71-gemma4-12b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `google/gemma-4-12B-it` | 0.983 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.961 ± 0.002 | 0.960 ± 0.001 | 0.962 ± 0.001 | ONNX int8 | 4.1 | 139.6 | 23 |

All latencies on the CPU with 2 threads.
