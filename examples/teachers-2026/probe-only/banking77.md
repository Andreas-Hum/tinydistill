| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77-7b-probe-only-zeroshot | zero-shot, 4-bit, probe only `Qwen/Qwen2.5-7B-Instruct` | 0.795 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.824 ± 0.002 | 0.822 ± 0.002 | 0.857 ± 0.004 | ONNX int8 | 1.2 | 38.4 | 23 |
| banking77-7b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `Qwen/Qwen2.5-7B-Instruct` | 0.854 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.870 ± 0.004 | 0.870 ± 0.004 | 0.892 ± 0.001 | ONNX int8 | 1.7 | 42.9 | 23 |

All latencies on the CPU with 2 threads.
