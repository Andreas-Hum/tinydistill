| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clinc150-gemma4-12b-similar16-probe | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.941 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.860 ± 0.005 | 0.898 ± 0.004 | 0.870 ± 0.004 | ONNX int8 | 0.8 | 22.4 | 23 |
| all-MiniLM-L12-v2 | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.941 | `sentence-transformers/all-MiniLM-L12-v2` | 33 | 0.872 ± 0.006 | 0.907 ± 0.003 | 0.880 ± 0.006 | ONNX int8 | 2.1 | 42.6 | 34 |
| all-mpnet-base-v2 | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.941 | `sentence-transformers/all-mpnet-base-v2` | 110 | 0.894 ± 0.005 | 0.921 ± 0.003 | 0.905 ± 0.003 | ONNX int8 | 7.2 | 131.2 | 110 |

All latencies on the CPU with 2 threads.
