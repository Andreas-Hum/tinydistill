| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| massive-da-gemma4-12b-probe-only-zeroshot | zero-shot, 4-bit, probe only `google/gemma-4-12B-it` | 0.760 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 24 | 0.779 ± 0.002 | 0.744 ± 0.003 | 0.820 ± 0.003 | ONNX int8 | 2.4 | 45.3 | 24 |
| massive-da-gemma4-12b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `google/gemma-4-12B-it` | 0.792 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 24 | 0.804 ± 0.002 | 0.777 ± 0.004 | 0.826 ± 0.003 | ONNX int8 | 1.6 | 40.6 | 24 |

All latencies on the CPU with 2 threads.
