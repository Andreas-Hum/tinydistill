| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77-gemma4-12b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `google/gemma-4-12B-it` | 0.873 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.870 ± 0.003 | 0.870 ± 0.003 | 0.906 ± 0.004 | ONNX int8 | 1.1 | 35.5 | 23 |
| banking77-qwen35-9b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `Qwen/Qwen3.5-9B` | 0.863 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.868 ± 0.003 | 0.867 ± 0.003 | 0.897 ± 0.002 | ONNX int8 | 1.0 | 36.2 | 23 |
| clinc150-gemma4-12b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `google/gemma-4-12B-it` | 0.925 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.852 ± 0.002 | 0.896 ± 0.004 | 0.859 ± 0.002 | ONNX int8 | 1.6 | 34.1 | 23 |
| clinc150-qwen35-9b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `Qwen/Qwen3.5-9B` | 0.899 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.841 ± 0.004 | 0.889 ± 0.003 | 0.856 ± 0.003 | ONNX int8 | 0.9 | 23.3 | 23 |
| massive-da-gemma4-12b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `google/gemma-4-12B-it` | 0.792 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 24 | 0.804 ± 0.002 | 0.777 ± 0.004 | 0.826 ± 0.003 | ONNX int8 | 1.6 | 40.6 | 24 |
| massive-da-qwen35-9b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `Qwen/Qwen3.5-9B` | 0.776 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 24 | 0.794 ± 0.002 | 0.764 ± 0.001 | 0.844 ± 0.001 | ONNX int8 | 1.8 | 40.6 | 24 |
| hwu64-gemma4-12b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `google/gemma-4-12B-it` | 0.879 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.891 ± 0.004 | 0.889 ± 0.004 | 0.897 ± 0.004 | ONNX int8 | 1.5 | 30.3 | 23 |
| hwu64-qwen35-9b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `Qwen/Qwen3.5-9B` | 0.887 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.888 ± 0.004 | 0.885 ± 0.006 | 0.897 ± 0.006 | ONNX int8 | 0.8 | 22.4 | 23 |

All latencies on the CPU with 2 threads.
