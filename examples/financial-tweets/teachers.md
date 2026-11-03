| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| financial-tweets-1.5b | zero-shot `Qwen/Qwen2.5-1.5B-Instruct` | 0.537 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.514 ± 0.004 | 0.538 ± 0.004 | 0.810 ± 0.004 | ONNX int8 | 1.6 | 85.1 | 23 |
| financial-tweets-1.5b-8shot | 8-shot `Qwen/Qwen2.5-1.5B-Instruct` | 0.314 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.295 ± 0.003 | 0.276 ± 0.003 | 0.870 ± 0.006 | ONNX int8 | 1.8 | 77.2 | 23 |
| financial-tweets-7b | zero-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.832 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.825 ± 0.007 | 0.768 ± 0.009 | 0.870 ± 0.005 | ONNX int8 | 2.5 | 105.7 | 23 |
| financial-tweets-ensemble | teacher ensemble of 3, zero-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3,microsoft/Phi-3.5-mini-instruct` | 0.842 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.828 ± 0.006 | 0.778 ± 0.005 | 0.884 ± 0.007 | ONNX int8 | 1.6 | 99.4 | 23 |
| financial-tweets-synthetic | zero-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.832 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.346 ± 0.049 | 0.324 ± 0.081 | 0.350 ± 0.043 | ONNX int8 | 1.3 | 102.3 | 23 |

All latencies on the CPU with 2 threads.
