| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ag-news-1.5b-plain3k | zero-shot `Qwen/Qwen2.5-1.5B-Instruct` | 0.755 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.785 ± 0.002 | 0.785 ± 0.002 | 0.825 ± 0.010 | ONNX int8 | 3.4 | 208.9 | 23 |
| ag-news-1.5b-reason3k | zero-shot, reasoning `Qwen/Qwen2.5-1.5B-Instruct` | 0.778 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.785 ± 0.004 | 0.777 ± 0.004 | 0.914 ± 0.003 | ONNX int8 | 3.5 | 208.3 | 23 |

All latencies on the CPU with 2 threads.
