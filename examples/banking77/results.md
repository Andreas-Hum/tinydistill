| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot | cpu | 0.406 | 0.367 | n/a | 13490.4 | n/a | 0.1 | 3087 | 1.0x |
| teacher, zero-shot | cuda | 0.406 | 0.367 | n/a | 264.9 | n/a | 3.8 | 3087 | 50.9x |
| student, PyTorch | cpu | 0.386 | 0.315 | 0.770 | 6.3 | 106.4 | 300.7 | 91 | 2127.4x |
| student, ONNX fp32 | cpu | 0.386 | 0.315 | 0.770 | 3.5 | 120.4 | 265.9 | 91 | 3909.0x |
| student, ONNX int8 | cpu | 0.386 | 0.316 | 0.763 | 2.2 | 49.1 | 651.2 | 23 | 6068.4x |

Task `banking77`, 10003 unlabelled training texts, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`.
CPU rows use 2 threads. Latency is the median over repeated runs. The teacher scores one text at a time, so it is timed at batch 1 and its throughput is 1 / latency. Speed-up is relative to the teacher, zero-shot on cpu.
