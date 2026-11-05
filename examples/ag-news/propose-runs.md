| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ag-news-1.5b | zero-shot `Qwen/Qwen2.5-1.5B-Instruct` | 0.749 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.775 ± 0.009 | 0.777 ± 0.009 | 0.830 ± 0.004 | ONNX int8 | 3.7 | 268.9 | 23 |
| ag-news-1.5b-proposed | zero-shot `Qwen/Qwen2.5-1.5B-Instruct` | 0.778 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.803 ± 0.002 | 0.806 ± 0.002 | 0.863 ± 0.002 | ONNX int8 | 4.2 | 309.6 | 23 |
| ag-news-1.5b-ensemble | zero-shot, 2 prompts `Qwen/Qwen2.5-1.5B-Instruct` | 0.770 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.796 ± 0.005 | 0.798 ± 0.005 | 0.852 ± 0.001 | ONNX int8 | 3.8 | 308.8 | 23 |

All latencies on the CPU with 2 threads.
