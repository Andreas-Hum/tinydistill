| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| financial-tweets-qwen35-9b-similar16 | 16 similar examples, 4-bit `Qwen/Qwen3.5-9B` | 0.821 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.805 ± 0.005 | 0.752 ± 0.006 | 0.836 ± 0.004 | ONNX int8 | 1.2 | 66.9 | 23 |
| financial-tweets-gemma4-12b-similar16 | 16 similar examples, 4-bit `google/gemma-4-12B-it` | 0.846 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.803 ± 0.006 | 0.766 ± 0.008 | 0.837 ± 0.003 | ONNX int8 | 1.1 | 67.5 | 23 |
| financial-tweets-ensemble-similar16 | teacher ensemble of 2, 16 similar examples, 4-bit `Qwen/Qwen3.5-9B,google/gemma-4-12B-it` | 0.859 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.827 ± 0.008 | 0.788 ± 0.008 | 0.853 ± 0.001 | ONNX int8 | 1.3 | 67.6 | 23 |

All latencies on the CPU with 2 threads.
