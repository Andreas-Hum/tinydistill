| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| massive-da-ensemble-similar16 | teacher ensemble of 2, 16 similar examples, 4-bit `Qwen/Qwen3.5-9B,google/gemma-4-12B-it` | 0.806 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.771 ± 0.003 | 0.710 ± 0.012 | 0.807 ± 0.003 | ONNX int8 | 0.8 | 29.1 | 23 |
| paraphrase-multilingual-MiniLM-L12-v2-trimmed | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.736 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 24 | 0.761 ± 0.001 | 0.708 ± 0.003 | 0.802 ± 0.006 | ONNX int8 | 1.8 | 40.2 | 24 |
| paraphrase-multilingual-MiniLM-L12-v2-trimmed-ensemble-similar16 | teacher ensemble of 2, 16 similar examples, 4-bit `Qwen/Qwen3.5-9B,google/gemma-4-12B-it` | 0.806 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 24 | 0.811 ± 0.002 | 0.783 ± 0.002 | 0.867 ± 0.003 | ONNX int8 | 1.9 | 46.2 | 24 |
| paraphrase-multilingual-MiniLM-L12-v2-trimmed-gemma4-12b-similar16-probe | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.808 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 24 | 0.815 ± 0.002 | 0.782 ± 0.005 | 0.849 ± 0.002 | ONNX int8 | 2.0 | 41.9 | 24 |

All latencies on the CPU with 2 threads.
