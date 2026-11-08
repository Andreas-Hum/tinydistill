| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe only | cuda | 0.854 | 0.854 | 0.943 | n/a | 77.3 | 1355.5 | 23.6 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.870 ± 0.004 | 0.870 ± 0.004 | 0.953 ± 0.004 | 0.892 ± 0.001 | 5.4 | 97.6 | 327.8 | 91 | 14.2x |
| student, ONNX fp32 | cpu | 0.870 | 0.870 | 0.956 | 0.892 | 3.4 | 89.2 | 358.6 | 91 | 22.6x |
| student, ONNX int8 | cpu | 0.869 | 0.868 | 0.956 | 0.888 | 1.7 | 42.9 | 745.5 | 23 | 45.6x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 616 gold training texts; fit on 493 of them, teacher accuracy on the other 123 is 75.6%. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 10003 training texts, plus cross-entropy on 616 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.156 | 0.869 | 0.868 | 1.7 |
| 10% | 0.602 | 0.875 | 0.874 | 9.4 |
| 20% | 0.761 | 0.869 | 0.869 | 17.2 |
| 30% | 0.854 | 0.865 | 0.864 | 24.9 |
| 50% | 0.924 | 0.858 | 0.857 | 40.4 |
| 100% | 1.000 | 0.854 | 0.854 | 79.0 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe only on cuda.
