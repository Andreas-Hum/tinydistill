| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| trec50-gemma4-12b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `google/gemma-4-12B-it` | 0.730 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.774 ± 0.016 | 0.622 ± 0.027 | 0.765 ± 0.008 | ONNX int8 | 0.8 | 23.8 | 23 |
| trec50-gemma4-12b-probe-only-zeroshot | zero-shot, 4-bit, probe only `google/gemma-4-12B-it` | 0.722 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.765 ± 0.004 | 0.632 ± 0.017 | 0.780 ± 0.009 | ONNX int8 | 0.8 | 24.0 | 23 |
| trec50-gemma4-12b-similar16 | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.706 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.741 ± 0.008 | 0.632 ± 0.007 | 0.763 ± 0.015 | ONNX int8 | 1.6 | 26.9 | 23 |

All latencies on the CPU with 2 threads.
