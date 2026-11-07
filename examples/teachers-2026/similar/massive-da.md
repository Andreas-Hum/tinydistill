| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| massive-da-qwen35-9b-similar16 | 16 similar examples, 4-bit `Qwen/Qwen3.5-9B` | 0.781 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.756 ± 0.002 | 0.700 ± 0.004 | 0.782 ± 0.002 | ONNX int8 | 0.9 | 30.4 | 23 |
| massive-da-gemma4-12b-similar16 | 16 similar examples, 4-bit `google/gemma-4-12B-it` | 0.798 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.760 ± 0.003 | 0.706 ± 0.009 | 0.796 ± 0.005 | ONNX int8 | 0.9 | 28.8 | 23 |
| massive-da-ensemble-similar16 | teacher ensemble of 2, 16 similar examples, 4-bit `Qwen/Qwen3.5-9B,google/gemma-4-12B-it` | 0.806 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.771 ± 0.003 | 0.710 ± 0.012 | 0.807 ± 0.003 | ONNX int8 | 0.8 | 29.1 | 23 |

All latencies on the CPU with 2 threads.
