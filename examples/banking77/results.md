| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot | cpu | 0.406 | 0.367 | n/a | 19955.9 | n/a | 0.1 | 6175 | 1.0x |
| teacher, zero-shot | cuda | 0.406 | 0.367 | n/a | 60.3 | n/a | 16.6 | 3087 | 331.1x |
| student, PyTorch | cpu | 0.386 | 0.315 | 0.772 | 3.9 | 78.3 | 408.9 | 91 | 5124.6x |
| student, ONNX fp32 | cpu | 0.386 | 0.315 | 0.772 | 2.6 | 82.4 | 388.4 | 91 | 7606.9x |
| student, ONNX int8 | cpu | 0.383 | 0.311 | 0.761 | 1.0 | 34.4 | 931.1 | 23 | 20502.9x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 10003 unlabelled training texts.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher scores one text at a time, so it is timed at batch 1 and its throughput is 1 / latency. Speed-up is relative to the teacher, zero-shot on cpu.
