| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.736 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118 | 0.761 ± 0.002 | 0.707 ± 0.003 | 0.804 ± 0.006 | ONNX int8 | 1.9 | 39.3 | 118 |
| paraphrase-multilingual-MiniLM-L12-v2-trimmed | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.736 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 24 | 0.761 ± 0.001 | 0.708 ± 0.003 | 0.802 ± 0.006 | ONNX int8 | 1.8 | 40.2 | 24 |

All latencies on the CPU with 2 threads.
