| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clinc150-gemma4-12b-similar16 | 16 similar examples, 4-bit `google/gemma-4-12B-it` | 0.914 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.850 ± 0.002 | 0.880 ± 0.002 | 0.863 ± 0.003 | ONNX int8 | 1.3 | 24.5 | 23 |
| clinc150-gemma4-12b-similar16-probe | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.941 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.860 ± 0.005 | 0.898 ± 0.004 | 0.870 ± 0.004 | ONNX int8 | 0.8 | 22.4 | 23 |

All latencies on the CPU with 2 threads.
