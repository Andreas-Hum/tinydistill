| Model | Device | Accuracy | Macro-F1 | Micro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit | cuda | 0.045 | 0.277 | 0.268 | n/a | 103.9 | 3016.5 | 10.6 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.008 ± 0.001 | 0.066 ± 0.001 | 0.137 ± 0.004 | 0.135 ± 0.000 | 5.8 | 113.1 | 283.0 | 91 | 17.9x |
| student, ONNX fp32 | cpu | 0.010 | 0.066 | 0.141 | 0.135 | 4.1 | 115.2 | 277.7 | 91 | 25.1x |
| student, ONNX int8 | cpu | 0.011 | 0.065 | 0.138 | 0.136 | 1.9 | 51.1 | 626.7 | 23 | 55.6x |
| baseline: student on 8 gold labels per class, no teacher | cpu | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.128 ± 0.000 | n/a | n/a | n/a | n/a | n/a |
| upper bound: student on all gold labels | cpu | 0.290 ± 0.016 | 0.118 ± 0.016 | 0.414 ± 0.021 | 0.066 ± 0.002 | n/a | n/a | n/a | n/a | n/a |

Task `go-emotions`, 5427 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns from the teacher's soft labels on all 10000 training texts, plus cross-entropy on 223 gold-labelled texts (8 per class; weight 1.0). The baseline is the same student trained with cross-entropy on those 223 gold labels only, without the teacher. The upper bound is the same student trained with cross-entropy on all 10000 gold training labels and no teacher; its speed equals the student's. Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 8, 10. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, 4-bit on cuda.
