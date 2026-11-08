| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clinc150-gemma4-12b-probe-only-zeroshot | zero-shot, 4-bit, probe only `google/gemma-4-12B-it` | 0.891 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.842 ± 0.004 | 0.889 ± 0.002 | 0.842 ± 0.005 | ONNX int8 | 1.0 | 26.8 | 23 |
| clinc150-gemma4-12b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `google/gemma-4-12B-it` | 0.925 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.852 ± 0.002 | 0.896 ± 0.004 | 0.859 ± 0.002 | ONNX int8 | 1.6 | 34.1 | 23 |

All latencies on the CPU with 2 threads.
