| Model | Device | Accuracy | Macro-F1 | Micro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot | cpu | 0.004 | 0.191 | 0.186 | n/a | 9990.9 | n/a | 0.1 | 6175 | 1.0x |
| teacher, zero-shot | cuda | 0.004 | 0.191 | 0.186 | n/a | 74.3 | 2131.6 | 15.0 | 3087 | 134.5x |
| student, PyTorch | cpu | 0.002 ± 0.001 | 0.111 ± 0.004 | 0.141 ± 0.002 | 0.015 ± 0.002 | 5.4 | 116.3 | 275.2 | 91 | 1857.3x |
| student, ONNX fp32 | cpu | 0.002 | 0.106 | 0.141 | 0.016 | 4.8 | 115.7 | 276.6 | 91 | 2072.9x |
| student, ONNX int8 | cpu | 0.002 | 0.108 | 0.141 | 0.016 | 2.0 | 66.3 | 482.8 | 23 | 5026.0x |
| baseline: student on 8 gold labels per class, no teacher | cpu | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.007 ± 0.000 | n/a | n/a | n/a | n/a | n/a |
| upper bound: student on all gold labels | cpu | 0.225 ± 0.021 | 0.034 ± 0.019 | 0.320 ± 0.034 | 0.007 ± 0.000 | n/a | n/a | n/a | n/a | n/a |

Task `go-emotions`, 5427 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns from the teacher's soft labels on all 10000 training texts, plus cross-entropy on 8 gold labels per class (223 texts, weight 1.0). The baseline is the same student trained with cross-entropy on those 223 gold labels only, without the teacher. The upper bound is the same student trained with cross-entropy on all 10000 gold training labels and no teacher; its speed equals the student's. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot on cpu.
