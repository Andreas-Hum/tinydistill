| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe only | cuda | 0.879 | 0.878 | 0.936 | n/a | 181.7 | 3464.1 | 9.2 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.891 ± 0.004 | 0.889 ± 0.004 | 0.960 ± 0.004 | 0.897 ± 0.004 | 7.2 | 66.7 | 480.1 | 91 | 25.2x |
| student, ONNX fp32 | cpu | 0.895 | 0.893 | 0.963 | 0.896 | 3.7 | 57.0 | 561.3 | 91 | 49.3x |
| student, ONNX int8 | cpu | 0.888 | 0.887 | 0.958 | 0.901 | 1.5 | 30.3 | 1054.6 | 23 | 121.4x |

Task `hwu64`, 1076 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 512 gold training texts; fit on 410 of them, teacher accuracy on the other 102 is 86.3%. The teacher's prompt shows each text the 16 most similar of 512 gold training examples. The student learns from the teacher's soft labels on all 8954 training texts, plus cross-entropy on 512 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.123 | 0.888 | 0.887 | 1.5 |
| 10% | 0.657 | 0.890 | 0.889 | 19.7 |
| 20% | 0.860 | 0.895 | 0.892 | 37.8 |
| 30% | 0.926 | 0.888 | 0.885 | 56.0 |
| 50% | 0.960 | 0.882 | 0.881 | 92.3 |
| 100% | 1.000 | 0.879 | 0.878 | 183.2 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe only on cuda.
