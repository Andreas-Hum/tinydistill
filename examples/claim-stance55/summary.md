| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| claim-stance55-gemma4-12b-probe-only-zeroshot | zero-shot, 4-bit, probe only `google/gemma-4-12B-it` | 0.896 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.844 ± 0.008 | 0.766 ± 0.018 | 0.837 ± 0.002 | ONNX int8 | 1.7 | 50.2 | 23 |
| claim-stance55-gemma4-12b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `google/gemma-4-12B-it` | 0.919 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.853 ± 0.005 | 0.776 ± 0.015 | 0.866 ± 0.006 | ONNX int8 | 2.4 | 57.8 | 23 |

All latencies on the CPU with 2 threads.
