| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77-7b-similar16 | 16 similar examples, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.851 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.857 ± 0.001 | 0.856 ± 0.001 | 0.878 ± 0.003 | ONNX int8 | 1.0 | 34.1 | 23 |
| banking77-7b-similar16-probe | 16 similar examples, 4-bit, probe `Qwen/Qwen2.5-7B-Instruct` | 0.859 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.875 ± 0.001 | 0.875 ± 0.001 | 0.890 ± 0.001 | ONNX int8 | 1.0 | 33.7 | 23 |

All latencies on the CPU with 2 threads.
