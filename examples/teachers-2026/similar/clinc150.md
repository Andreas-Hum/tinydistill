| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clinc150-qwen35-9b-similar16 | 16 similar examples, 4-bit `Qwen/Qwen3.5-9B` | 0.841 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.805 ± 0.005 | 0.848 ± 0.004 | 0.822 ± 0.003 | ONNX int8 | 1.0 | 23.7 | 23 |
| clinc150-gemma4-12b-similar16 | 16 similar examples, 4-bit `google/gemma-4-12B-it` | 0.914 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.850 ± 0.002 | 0.880 ± 0.002 | 0.863 ± 0.003 | ONNX int8 | 1.3 | 24.5 | 23 |
| clinc150-ensemble-similar16 | teacher ensemble of 2, 16 similar examples, 4-bit `Qwen/Qwen3.5-9B,google/gemma-4-12B-it` | 0.914 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.836 ± 0.001 | 0.873 ± 0.001 | 0.856 ± 0.001 | ONNX int8 | 1.1 | 24.7 | 23 |

All latencies on the CPU with 2 threads.
