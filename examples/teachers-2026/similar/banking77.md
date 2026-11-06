| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77-qwen35-9b-similar16 | 16 similar examples, 4-bit `Qwen/Qwen3.5-9B` | 0.861 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.867 ± 0.004 | 0.867 ± 0.004 | 0.905 ± 0.001 | ONNX int8 | 0.9 | 33.0 | 23 |
| banking77-gemma4-12b-similar16 | 16 similar examples, 4-bit `google/gemma-4-12B-it` | 0.870 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.870 ± 0.005 | 0.869 ± 0.004 | 0.913 ± 0.001 | ONNX int8 | 1.2 | 37.5 | 23 |
| banking77-ensemble-similar16 | teacher ensemble of 2, 16 similar examples, 4-bit `Qwen/Qwen3.5-9B,google/gemma-4-12B-it` | 0.879 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.875 ± 0.003 | 0.875 ± 0.003 | 0.924 ± 0.001 | ONNX int8 | 1.4 | 40.6 | 23 |

All latencies on the CPU with 2 threads.
