| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit | cuda | 0.564 | 0.614 | 0.731 | n/a | 261.7 | 8425.4 | 3.8 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.565 ± 0.002 | 0.608 ± 0.001 | 0.734 ± 0.004 | 0.726 ± 0.003 | 4.4 | 56.0 | 571.5 | 91 | 60.1x |
| student, ONNX fp32 | cpu | 0.565 | 0.608 | 0.729 | 0.722 | 2.7 | 52.0 | 615.6 | 91 | 95.5x |
| student, ONNX int8 | cpu | 0.559 | 0.602 | 0.730 | 0.717 | 1.3 | 24.3 | 1316.5 | 23 | 200.9x |
| upper bound: student on all gold labels | cpu | 0.870 ± 0.001 | 0.907 ± 0.002 | 0.913 ± 0.004 | 0.569 ± 0.002 | n/a | n/a | n/a | n/a | n/a |

Task `clinc150`, 5500 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 15250 unlabelled training texts. The upper bound is the same student trained with cross-entropy on all 15250 gold training labels and no teacher; its speed equals the student's. Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 5, 7, 8, 10. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.074 | 0.559 | 0.602 | 1.3 |
| 10% | 0.350 | 0.565 | 0.609 | 27.5 |
| 20% | 0.505 | 0.571 | 0.614 | 53.6 |
| 30% | 0.657 | 0.574 | 0.617 | 79.8 |
| 50% | 0.889 | 0.569 | 0.617 | 132.2 |
| 100% | 1.000 | 0.564 | 0.614 | 263.0 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot, 4-bit on cuda.
