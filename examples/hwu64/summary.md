| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hwu64-gemma4-12b-probe-only-zeroshot | zero-shot, 4-bit, probe only `google/gemma-4-12B-it` | 0.842 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.865 ± 0.001 | 0.865 ± 0.004 | 0.880 ± 0.003 | ONNX int8 | 1.1 | 30.9 | 23 |
| hwu64-gemma4-12b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `google/gemma-4-12B-it` | 0.879 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.891 ± 0.004 | 0.889 ± 0.004 | 0.897 ± 0.004 | ONNX int8 | 1.5 | 30.3 | 23 |

All latencies on the CPU with 2 threads.
