| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ap106-gemma4-12b-probe-only-zeroshot | zero-shot, 4-bit, probe only `google/gemma-4-12B-it` | 0.410 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.435 ± 0.001 | 0.378 ± 0.010 | 0.860 ± 0.003 | ONNX int8 | 3.8 | 244.4 | 23 |
| ap106-gemma4-12b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `google/gemma-4-12B-it` | 0.400 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.433 ± 0.002 | 0.383 ± 0.008 | 0.836 ± 0.007 | ONNX int8 | 2.9 | 161.5 | 23 |

All latencies on the CPU with 2 threads.
