| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe only | cuda | 0.983 | 0.983 | 0.988 | n/a | 412.9 | 8930.6 | 3.6 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.961 ± 0.002 | 0.960 ± 0.001 | 0.978 ± 0.002 | 0.962 ± 0.001 | 18.6 | 299.4 | 106.9 | 91 | 22.2x |
| student, ONNX fp32 | cpu | 0.963 | 0.962 | 0.980 | 0.962 | 10.0 | 289.4 | 110.6 | 91 | 41.3x |
| student, ONNX int8 | cpu | 0.963 | 0.961 | 0.979 | 0.962 | 4.1 | 139.6 | 229.2 | 23 | 101.3x |

Task `argument-topic71`, 1898 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 568 gold training texts; fit on 454 of them, teacher accuracy on the other 114 is 94.7%. The teacher's prompt shows each text the 16 most similar of 568 gold training examples. The student learns from the teacher's soft labels on all 6640 training texts, plus cross-entropy on 568 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.118 | 0.963 | 0.961 | 4.1 |
| 10% | 0.968 | 0.982 | 0.982 | 45.4 |
| 20% | 0.981 | 0.983 | 0.983 | 86.8 |
| 30% | 0.983 | 0.983 | 0.983 | 127.9 |
| 50% | 0.986 | 0.983 | 0.983 | 210.5 |
| 100% | 1.000 | 0.983 | 0.983 | 417.0 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe only on cuda.
