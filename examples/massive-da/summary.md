| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| massive-da-7b | zero-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.603 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.594 ± 0.007 | 0.536 ± 0.008 | 0.725 ± 0.003 | ONNX int8 | 0.8 | 29.6 | 23 |
| massive-da-7b-8shot | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.736 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.724 ± 0.006 | 0.667 ± 0.004 | 0.752 ± 0.006 | ONNX int8 | 0.9 | 29.5 | 23 |
| paraphrase-multilingual-MiniLM-L12-v2 | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.736 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118 | 0.761 ± 0.002 | 0.707 ± 0.003 | 0.804 ± 0.006 | ONNX int8 | 1.9 | 39.3 | 118 |

All latencies on the CPU with 2 threads.
