| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit | cuda | 0.630 | 0.607 | n/a | 295.4 | n/a | 3.4 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.627 | 0.583 | 0.765 | 4.5 | 93.6 | 341.9 | 91 | 65.5x |
| upper bound: student on all gold labels | cpu | 0.911 | 0.909 | 0.631 | n/a | n/a | n/a | n/a | n/a |
| student, ONNX fp32 | cpu | 0.627 | 0.583 | 0.765 | 3.2 | 80.9 | 395.7 | 91 | 92.5x |
| student, ONNX int8 | cpu | 0.622 | 0.576 | 0.762 | 1.1 | 40.7 | 786.4 | 23 | 262.6x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns from the teacher's soft labels on all 10003 training texts, plus cross-entropy on 8 gold labels per class (616 texts, weight 1.0). The upper bound is the same student trained with cross-entropy on all 10003 gold training labels and no teacher; its speed equals the student's.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher scores one text at a time, so it is timed at batch 1 and its throughput is 1 / latency. Speed-up is relative to the teacher, zero-shot, 4-bit on cuda.
