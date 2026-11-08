| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit, probe only | cuda | 0.795 | 0.792 | 0.920 | n/a | 37.3 | 155.8 | 205.3 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.824 ± 0.002 | 0.822 ± 0.002 | 0.938 ± 0.002 | 0.857 ± 0.004 | 5.1 | 84.8 | 377.3 | 91 | 7.3x |
| student, ONNX fp32 | cpu | 0.826 | 0.824 | 0.936 | 0.856 | 2.9 | 81.6 | 392.2 | 91 | 12.7x |
| student, ONNX int8 | cpu | 0.824 | 0.823 | 0.934 | 0.850 | 1.2 | 38.4 | 833.5 | 23 | 30.2x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 616 gold training texts; fit on 493 of them, teacher accuracy on the other 123 is 75.6%. The student learns from the teacher's soft labels on all 10003 training texts, plus cross-entropy on 616 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.124 | 0.824 | 0.823 | 1.2 |
| 10% | 0.513 | 0.827 | 0.823 | 5.0 |
| 20% | 0.675 | 0.821 | 0.818 | 8.7 |
| 30% | 0.789 | 0.816 | 0.812 | 12.4 |
| 50% | 0.899 | 0.802 | 0.799 | 19.9 |
| 100% | 1.000 | 0.795 | 0.792 | 38.6 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot, 4-bit, probe only on cuda.
