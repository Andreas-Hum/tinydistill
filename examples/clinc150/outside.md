| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clinc150-7b-shortlist20-keep-oos | zero-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.581 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.573 ± 0.003 | 0.604 ± 0.005 | 0.724 ± 0.003 | ONNX int8 | 1.6 | 29.0 | 23 |
| clinc150-7b-outside | zero-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.575 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.567 ± 0.007 | 0.594 ± 0.004 | 0.719 ± 0.005 | ONNX int8 | 1.0 | 22.7 | 23 |
| clinc150-7b-outside-similar16 | 16 similar examples, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.769 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.767 ± 0.002 | 0.832 ± 0.001 | 0.791 ± 0.001 | ONNX int8 | 1.6 | 42.5 | 23 |

All latencies on the CPU with 2 threads.
