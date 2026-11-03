| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77-7b-8shot | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.729 ± 0.011 | 0.697 ± 0.015 | 0.800 ± 0.003 | ONNX int8 | 1.4 | 40.2 | 23 |
| all-MiniLM-L12-v2 | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `sentence-transformers/all-MiniLM-L12-v2` | 33 | 0.743 ± 0.011 | 0.710 ± 0.016 | 0.815 ± 0.004 | ONNX int8 | 3.3 | 79.2 | 34 |
| all-MiniLM-L12-v2-20-epochs | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `sentence-transformers/all-MiniLM-L12-v2` | 33 | 0.817 ± 0.001 | 0.814 ± 0.001 | 0.875 ± 0.002 | ONNX int8 | 3.2 | 82.0 | 34 |
| all-MiniLM-L6-v2-20-epochs | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.813 ± 0.002 | 0.811 ± 0.003 | 0.868 ± 0.003 | ONNX int8 | 1.2 | 36.3 | 23 |
| all-mpnet-base-v2 | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `sentence-transformers/all-mpnet-base-v2` | 110 | 0.814 ± 0.004 | 0.805 ± 0.007 | 0.863 ± 0.001 | ONNX int8 | 7.3 | 220.4 | 110 |
| bge-small-en-v1.5 | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `BAAI/bge-small-en-v1.5` | 33 | 0.739 ± 0.008 | 0.706 ± 0.011 | 0.815 ± 0.004 | ONNX int8 | 2.7 | 75.2 | 34 |
| static | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `static` | 8 | 0.803 ± 0.001 | 0.801 ± 0.001 | 0.824 ± 0.001 | ONNX int8 | 0.1 | 5.1 | 8 |

All latencies on the CPU with 2 threads.
