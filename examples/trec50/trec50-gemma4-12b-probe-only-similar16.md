| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe only | cuda | 0.730 | 0.585 | 0.888 | n/a | 145.4 | 3832.2 | 8.4 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.774 ± 0.016 | 0.622 ± 0.027 | 0.921 ± 0.004 | 0.765 ± 0.008 | 3.7 | 50.7 | 631.6 | 91 | 39.7x |
| student, ONNX fp32 | cpu | 0.786 | 0.591 | 0.920 | 0.756 | 2.2 | 48.0 | 667.3 | 91 | 65.2x |
| student, ONNX int8 | cpu | 0.780 | 0.605 | 0.926 | 0.758 | 0.8 | 23.8 | 1343.1 | 23 | 184.7x |

Task `trec50`, 500 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 381 gold training texts; fit on 305 of them, teacher accuracy on the other 76 is 71.1%. The teacher's prompt shows each text the 16 most similar of 390 gold training examples. The student learns from the teacher's soft labels on all 5452 training texts, plus cross-entropy on 390 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.111 | 0.780 | 0.605 | 0.8 |
| 10% | 0.315 | 0.788 | 0.610 | 15.3 |
| 20% | 0.423 | 0.804 | 0.642 | 29.9 |
| 30% | 0.529 | 0.792 | 0.639 | 44.4 |
| 50% | 0.719 | 0.756 | 0.605 | 73.5 |
| 100% | 1.000 | 0.730 | 0.585 | 146.2 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe only on cuda.
