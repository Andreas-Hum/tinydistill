| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe only | cuda | 0.919 | 0.864 | 0.965 | n/a | 203.2 | 5336.7 | 6.0 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.853 ± 0.005 | 0.776 ± 0.015 | 0.931 ± 0.000 | 0.866 ± 0.006 | 6.0 | 114.8 | 278.6 | 91 | 33.6x |
| student, ONNX fp32 | cpu | 0.852 | 0.791 | 0.931 | 0.865 | 4.1 | 106.4 | 300.9 | 91 | 49.6x |
| student, ONNX int8 | cpu | 0.856 | 0.799 | 0.931 | 0.867 | 2.4 | 57.8 | 553.9 | 23 | 85.4x |

Task `claim-stance55`, 480 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 422 gold training texts; fit on 338 of them, teacher accuracy on the other 84 is 92.9%. The teacher's prompt shows each text the 16 most similar of 424 gold training examples. The student learns from the teacher's soft labels on all 1675 training texts, plus cross-entropy on 424 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.061 | 0.856 | 0.799 | 2.4 |
| 10% | 0.164 | 0.890 | 0.857 | 22.7 |
| 20% | 0.272 | 0.904 | 0.873 | 43.0 |
| 30% | 0.327 | 0.906 | 0.861 | 63.3 |
| 50% | 0.556 | 0.919 | 0.866 | 104.0 |
| 100% | 1.000 | 0.919 | 0.864 | 205.6 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe only on cuda.
