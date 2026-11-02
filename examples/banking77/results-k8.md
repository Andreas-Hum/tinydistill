| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot | cpu | 0.406 | 0.367 | n/a | 19699.4 | n/a | 0.1 | 6175 | 1.0x |
| teacher, zero-shot | cuda | 0.406 | 0.367 | n/a | 58.3 | n/a | 17.2 | 3087 | 338.1x |
| student, PyTorch | cpu | 0.404 | 0.324 | 0.699 | 3.9 | 76.7 | 417.4 | 91 | 5056.3x |
| upper bound: student on all gold labels | cpu | 0.911 | 0.909 | 0.412 | n/a | n/a | n/a | n/a | n/a |
| student, ONNX fp32 | cpu | 0.404 | 0.324 | 0.699 | 2.7 | 72.1 | 443.6 | 91 | 7306.9x |
| student, ONNX int8 | cpu | 0.398 | 0.318 | 0.697 | 1.0 | 36.7 | 872.1 | 23 | 18792.2x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns from the teacher's soft labels on all 10003 training texts, plus cross-entropy on 8 gold labels per class (616 texts, weight 1.0). The upper bound is the same student trained with cross-entropy on all 10003 gold training labels and no teacher; its speed equals the student's.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher scores one text at a time, so it is timed at batch 1 and its throughput is 1 / latency. Speed-up is relative to the teacher, zero-shot on cpu.
