| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77-7b-similar16-probe | 16 similar examples, 4-bit, probe `Qwen/Qwen2.5-7B-Instruct` | 0.859 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.875 ± 0.001 | 0.875 ± 0.001 | 0.890 ± 0.001 | ONNX int8 | 1.0 | 33.7 | 23 |
| banking77-mpnet | 16 similar examples, 4-bit, probe `Qwen/Qwen2.5-7B-Instruct` | 0.859 | `sentence-transformers/all-mpnet-base-v2` | 110 | 0.878 ± 0.002 | 0.878 ± 0.002 | 0.898 ± 0.002 | ONNX int8 | 6.9 | 200.1 | 111 |
| hwu64-gemma4-12b-similar16 | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.877 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.894 ± 0.005 | 0.893 ± 0.004 | 0.893 ± 0.003 | ONNX int8 | 0.8 | 21.7 | 23 |
| hwu64-mpnet | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.877 | `sentence-transformers/all-mpnet-base-v2` | 110 | 0.891 ± 0.002 | 0.891 ± 0.001 | 0.904 ± 0.002 | ONNX int8 | 6.1 | 130.9 | 111 |
| paraphrase-multilingual-MiniLM-L12-v2-trimmed-gemma4-12b-similar16-probe | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.808 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 24 | 0.815 ± 0.002 | 0.782 ± 0.005 | 0.849 ± 0.002 | ONNX int8 | 2.0 | 41.9 | 24 |
| massive-da-multilingual-mpnet | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.808 | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 90 | 0.818 ± 0.004 | 0.793 ± 0.004 | 0.861 ± 0.003 | ONNX int8 | 7.4 | 142.7 | 92 |

All latencies on the CPU with 2 threads.
