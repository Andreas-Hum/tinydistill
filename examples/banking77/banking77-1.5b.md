| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot | cpu | 0.404 | 0.365 | n/a | 11701.8 | n/a | 0.1 | 6175 | 1.0x |
| teacher, zero-shot | cuda | 0.404 | 0.365 | n/a | 46.8 | 1321.5 | 24.2 | 3087 | 250.2x |
| student, PyTorch | cpu | 0.386 ± 0.004 | 0.313 ± 0.005 | 0.766 ± 0.004 | 4.4 | 81.6 | 392.3 | 91 | 2645.3x |
| student, ONNX fp32 | cpu | 0.386 | 0.315 | 0.768 | 3.1 | 80.6 | 396.9 | 91 | 3727.2x |
| student, ONNX int8 | cpu | 0.385 | 0.315 | 0.765 | 1.2 | 37.3 | 858.8 | 23 | 9938.4x |
| upper bound: student on all gold labels | cpu | 0.914 ± 0.005 | 0.914 ± 0.005 | 0.410 ± 0.003 | n/a | n/a | n/a | n/a | n/a |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 10003 unlabelled training texts. The upper bound is the same student trained with cross-entropy on all 10003 gold training labels and no teacher; its speed equals the student's. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot on cpu.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.078 | 0.385 | 0.315 | 1.2 |
| 10% | 0.356 | 0.397 | 0.341 | 5.9 |
| 20% | 0.459 | 0.401 | 0.349 | 10.5 |
| 30% | 0.533 | 0.409 | 0.359 | 15.2 |
| 50% | 0.696 | 0.403 | 0.360 | 24.6 |
| 100% | 1.000 | 0.404 | 0.365 | 47.9 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot on cuda.
