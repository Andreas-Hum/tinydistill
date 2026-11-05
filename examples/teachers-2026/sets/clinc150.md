| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clinc150-7b-similar16 | 16 similar examples, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.775 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.767 ± 0.006 | 0.834 ± 0.003 | 0.796 ± 0.004 | ONNX int8 | 0.8 | 22.2 | 23 |
| clinc150-7b-similar16-sets3 | 16 similar examples, 3 example sets, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.702 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.749 ± 0.005 | 0.816 ± 0.003 | 0.734 ± 0.004 | ONNX int8 | 0.9 | 23.0 | 23 |

All latencies on the CPU with 2 threads.
