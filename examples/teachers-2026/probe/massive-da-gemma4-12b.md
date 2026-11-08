| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| massive-da-gemma4-12b-similar16 | 16 similar examples, 4-bit `google/gemma-4-12B-it` | 0.798 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.760 ± 0.003 | 0.706 ± 0.009 | 0.796 ± 0.005 | ONNX int8 | 0.9 | 28.8 | 23 |
| massive-da-gemma4-12b-similar16-probe | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.808 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.768 ± 0.004 | 0.721 ± 0.003 | 0.797 ± 0.003 | ONNX int8 | 1.1 | 31.9 | 23 |

All latencies on the CPU with 2 threads.
