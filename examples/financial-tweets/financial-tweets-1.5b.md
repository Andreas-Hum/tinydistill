| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot | cpu | 0.537 | 0.558 | n/a | 1053.7 | 33390.7 | 1.0 | 6175 | 1.0x |
| teacher, zero-shot | cuda | 0.537 | 0.558 | n/a | 32.2 | 140.9 | 227.2 | 3087 | 32.8x |
| student, PyTorch | cpu | 0.514 ± 0.004 | 0.538 ± 0.004 | 0.810 ± 0.004 | 5.9 | 167.7 | 190.8 | 91 | 178.8x |
| student, ONNX fp32 | cpu | 0.518 | 0.542 | 0.810 | 3.9 | 169.5 | 188.8 | 91 | 273.0x |
| student, ONNX int8 | cpu | 0.513 | 0.539 | 0.805 | 1.6 | 85.1 | 376.2 | 23 | 674.6x |
| upper bound: student on all gold labels | cpu | 0.874 ± 0.005 | 0.832 ± 0.007 | 0.531 ± 0.004 | n/a | n/a | n/a | n/a | n/a |

Task `financial-tweets`, 2388 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 9543 unlabelled training texts. The upper bound is the same student trained with cross-entropy on all 9543 gold training labels and no teacher; its speed equals the student's. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot on cpu.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.367 | 0.513 | 0.539 | 1.6 |
| 10% | 0.553 | 0.521 | 0.547 | 4.8 |
| 20% | 0.630 | 0.528 | 0.553 | 8.0 |
| 30% | 0.706 | 0.529 | 0.552 | 11.2 |
| 50% | 0.856 | 0.528 | 0.551 | 17.6 |
| 100% | 1.000 | 0.537 | 0.558 | 33.7 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot on cuda.
