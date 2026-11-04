| Model | Device | Accuracy | Macro-F1 | Micro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit | cuda:0 | 0.045 | 0.277 | 0.268 | n/a | n/a | n/a | n/a | n/a | n/a |
| student, PyTorch | cpu | 0.057 ± 0.005 | 0.226 ± 0.017 | 0.253 ± 0.003 | 0.157 ± 0.005 | 5.9 | 107.4 | 297.9 | 91 | n/a |
| student, ONNX fp32 | cpu | 0.060 | 0.208 | 0.250 | 0.161 | 3.7 | 103.2 | 310.0 | 91 | n/a |
| student, ONNX int8 | cpu | 0.059 | 0.199 | 0.246 | 0.160 | 1.6 | 50.1 | 639.3 | 23 | n/a |
| baseline: student on 8 gold labels per class, no teacher | cpu | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.128 ± 0.000 | n/a | n/a | n/a | n/a | n/a |
| upper bound: student on all gold labels | cpu | 0.308 ± 0.009 | 0.134 ± 0.003 | 0.431 ± 0.011 | 0.063 ± 0.002 | n/a | n/a | n/a | n/a | n/a |

Task `go-emotions`, 5427 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns from the teacher's soft labels on all 10000 training texts, plus cross-entropy on 223 gold-labelled texts (8 per class; weight 1.0). The baseline is the same student trained with cross-entropy on those 223 gold labels only, without the teacher. The upper bound is the same student trained with cross-entropy on all 10000 gold training labels and no teacher; its speed equals the student's. Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 30 epochs, stopping after 3 without improvement); kept epochs: 9, 10, 11, 12, 15, 16, 18, 22. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. 
