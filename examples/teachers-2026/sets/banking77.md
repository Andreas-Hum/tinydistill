| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77-7b-similar16 | 16 similar examples, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.851 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.857 ± 0.001 | 0.856 ± 0.001 | 0.878 ± 0.003 | ONNX int8 | 1.0 | 34.1 | 23 |
| banking77-7b-similar16-sets3 | 16 similar examples, 3 example sets, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.749 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.809 ± 0.002 | 0.801 ± 0.002 | 0.795 ± 0.003 | ONNX int8 | 0.9 | 34.0 | 23 |

All latencies on the CPU with 2 threads.
