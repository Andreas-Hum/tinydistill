| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit | cuda | 0.630 | 0.607 | n/a | 272.2 | n/a | 3.7 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.623 | 0.587 | 0.827 | 5.5 | 93.0 | 344.2 | 91 | 49.4x |
| student, ONNX fp32 | cpu | 0.623 | 0.587 | 0.827 | 3.4 | 86.8 | 368.7 | 91 | 80.8x |
| student, ONNX int8 | cpu | 0.622 | 0.586 | 0.822 | 1.6 | 43.8 | 730.0 | 23 | 171.2x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 10003 unlabelled training texts.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher scores one text at a time, so it is timed at batch 1 and its throughput is 1 / latency. Speed-up is relative to the teacher, zero-shot, 4-bit on cuda.
