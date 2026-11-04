| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| imdb-truncated | zero-shot `Qwen/Qwen2.5-1.5B-Instruct` | 0.934 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.844 ± 0.007 | 0.843 ± 0.007 | 0.851 ± 0.005 | ONNX int8 | 6.4 | 217.4 | 23 |
| imdb-chunked | zero-shot `Qwen/Qwen2.5-1.5B-Instruct` | 0.875 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.860 ± 0.006 | 0.858 ± 0.006 | 0.926 ± 0.001 | ONNX int8 | 18.0 | 866.0 | 23 |

All latencies on the CPU with 2 threads.
