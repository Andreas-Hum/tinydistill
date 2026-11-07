| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.846 | 0.823 | n/a | 342.1 | 10041.1 | 3.2 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.803 ± 0.006 | 0.766 ± 0.008 | 0.837 ± 0.003 | 4.1 | 138.1 | 231.7 | 91 | 83.5x |
| student, ONNX fp32 | cpu | 0.800 | 0.761 | 0.836 | 2.5 | 139.1 | 230.0 | 91 | 134.6x |
| student, ONNX int8 | cpu | 0.791 | 0.752 | 0.827 | 1.1 | 67.5 | 473.8 | 23 | 305.7x |

Task `financial-tweets`, 2388 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 24 gold training examples. The student learns from the teacher's soft labels on all 9543 training texts, plus cross-entropy on 24 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.365 | 0.791 | 0.752 | 1.1 |
| 10% | 0.751 | 0.817 | 0.782 | 35.4 |
| 20% | 0.900 | 0.824 | 0.792 | 69.6 |
| 30% | 0.962 | 0.834 | 0.806 | 103.7 |
| 50% | 0.989 | 0.846 | 0.822 | 172.2 |
| 100% | 1.000 | 0.846 | 0.823 | 343.3 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit on cuda.
