| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clinc150-7b-full | zero-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.564 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.565 ± 0.002 | 0.608 ± 0.001 | 0.726 ± 0.003 | ONNX int8 | 1.3 | 24.3 | 23 |
| clinc150-7b-shortlist20 | zero-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.555 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.556 ± 0.001 | 0.600 ± 0.002 | 0.726 ± 0.002 | ONNX int8 | 1.7 | 28.2 | 23 |
| clinc150-7b-shortlist20-keep-oos | zero-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.581 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.573 ± 0.003 | 0.604 ± 0.005 | 0.724 ± 0.003 | ONNX int8 | 1.6 | 29.0 | 23 |

All latencies on the CPU with 2 threads.
