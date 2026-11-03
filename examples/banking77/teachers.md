| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77-1.5b | zero-shot `Qwen/Qwen2.5-1.5B-Instruct` | 0.404 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.386 ± 0.004 | 0.313 ± 0.005 | 0.766 ± 0.004 | ONNX int8 | 1.2 | 37.3 | 23 |
| banking77-1.5b-8shot | 8-shot `Qwen/Qwen2.5-1.5B-Instruct` | 0.513 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.485 ± 0.009 | 0.406 ± 0.007 | 0.706 ± 0.002 | ONNX int8 | 1.1 | 36.4 | 23 |
| banking77-7b | zero-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.629 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.618 ± 0.003 | 0.579 ± 0.005 | 0.817 ± 0.007 | ONNX int8 | 1.4 | 39.2 | 23 |
| banking77-7b-8shot | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.729 ± 0.011 | 0.697 ± 0.015 | 0.800 ± 0.003 | ONNX int8 | 1.4 | 40.2 | 23 |

All latencies on the CPU with 2 threads.
