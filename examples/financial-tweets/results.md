| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot | cpu | 0.535 | 0.557 | n/a | 976.3 | n/a | 1.0 | 6175 | 1.0x |
| teacher, zero-shot | cuda | 0.535 | 0.557 | n/a | 33.7 | n/a | 29.6 | 3087 | 28.9x |
| student, PyTorch | cpu | 0.517 | 0.542 | 0.815 | 4.5 | 145.4 | 220.1 | 91 | 219.3x |
| upper bound: student on all gold labels | cpu | 0.867 | 0.824 | 0.531 | n/a | n/a | n/a | n/a | n/a |
| student, ONNX fp32 | cpu | 0.517 | 0.542 | 0.815 | 3.2 | 148.5 | 215.4 | 91 | 303.9x |
| student, ONNX int8 | cpu | 0.513 | 0.538 | 0.810 | 1.3 | 71.5 | 447.7 | 23 | 754.7x |

Task `financial-tweets`, 2388 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 9543 unlabelled training texts. The upper bound is the same student trained with cross-entropy on all 9543 gold training labels and no teacher; its speed equals the student's.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher scores one text at a time, so it is timed at batch 1 and its throughput is 1 / latency. Speed-up is relative to the teacher, zero-shot on cpu.
