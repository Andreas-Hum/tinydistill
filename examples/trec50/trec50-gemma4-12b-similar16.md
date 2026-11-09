| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe | cuda | 0.706 | 0.633 | 0.910 | n/a | 2837.2 | n/a | 0.4 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.741 ± 0.008 | 0.632 ± 0.007 | 0.921 ± 0.008 | 0.763 ± 0.015 | 5.9 | 64.4 | 496.7 | 91 | 482.2x |
| student, ONNX fp32 | cpu | 0.736 | 0.624 | 0.928 | 0.750 | 3.3 | 58.5 | 547.2 | 91 | 858.8x |
| student, ONNX int8 | cpu | 0.740 | 0.622 | 0.924 | 0.742 | 1.6 | 26.9 | 1188.6 | 23 | 1800.3x |

Task `trec50`, 500 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's probabilities are averaged with those of a linear probe on its hidden states, fit on 381 gold training texts; fit on 305 of them, teacher accuracy on the other 76 is 73.7% in context, 71.1% for the probe and 69.7% combined. The teacher's prompt shows each text the 16 most similar of 390 gold training examples. The student learns from the teacher's soft labels on all 5452 training texts, plus cross-entropy on 390 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.172 | 0.740 | 0.622 | 1.6 |
| 10% | 0.329 | 0.754 | 0.642 | 285.3 |
| 20% | 0.430 | 0.762 | 0.645 | 569.0 |
| 30% | 0.527 | 0.740 | 0.643 | 852.7 |
| 50% | 0.703 | 0.722 | 0.640 | 1420.2 |
| 100% | 1.000 | 0.706 | 0.633 | 2838.7 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe on cuda.
