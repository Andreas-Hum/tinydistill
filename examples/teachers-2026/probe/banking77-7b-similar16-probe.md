| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe | cuda | 0.859 | 0.859 | 0.946 | n/a | 155.4 | 4910.8 | 6.5 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.875 ± 0.001 | 0.875 ± 0.001 | 0.953 ± 0.003 | 0.890 ± 0.001 | 3.7 | 73.0 | 438.6 | 91 | 41.6x |
| student, ONNX fp32 | cpu | 0.875 | 0.875 | 0.954 | 0.889 | 2.6 | 71.8 | 445.5 | 91 | 60.7x |
| student, ONNX int8 | cpu | 0.872 | 0.871 | 0.955 | 0.885 | 1.0 | 33.7 | 948.8 | 23 | 152.9x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's probabilities are averaged with those of a linear probe on its hidden states, fit on 616 gold training texts; fit on 493 of them, teacher accuracy on the other 123 is 78.0% in context, 75.6% for the probe and 78.9% combined. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 10003 training texts, plus cross-entropy on 616 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.125 | 0.872 | 0.871 | 1.0 |
| 10% | 0.596 | 0.877 | 0.877 | 16.6 |
| 20% | 0.768 | 0.874 | 0.873 | 32.1 |
| 30% | 0.856 | 0.868 | 0.867 | 47.6 |
| 50% | 0.925 | 0.861 | 0.860 | 78.7 |
| 100% | 1.000 | 0.859 | 0.859 | 156.4 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe on cuda.
