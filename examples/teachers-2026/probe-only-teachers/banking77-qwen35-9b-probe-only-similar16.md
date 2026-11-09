| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe only | cuda | 0.863 | 0.862 | 0.956 | n/a | 103.7 | 2363.1 | 13.5 | 7530 | 1.0x |
| student, PyTorch | cpu | 0.868 ± 0.003 | 0.867 ± 0.003 | 0.960 ± 0.002 | 0.897 ± 0.002 | 4.0 | 79.4 | 403.2 | 91 | 25.7x |
| student, ONNX fp32 | cpu | 0.867 | 0.866 | 0.958 | 0.894 | 2.9 | 78.4 | 408.4 | 91 | 36.0x |
| student, ONNX int8 | cpu | 0.866 | 0.865 | 0.956 | 0.892 | 1.0 | 36.2 | 883.3 | 23 | 101.0x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen3.5-9B`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 616 gold training texts; fit on 493 of them, teacher accuracy on the other 123 is 77.2%. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 10003 training texts, plus cross-entropy on 616 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.147 | 0.866 | 0.865 | 1.0 |
| 10% | 0.620 | 0.875 | 0.874 | 11.4 |
| 20% | 0.803 | 0.875 | 0.873 | 21.8 |
| 30% | 0.879 | 0.870 | 0.868 | 32.1 |
| 50% | 0.935 | 0.867 | 0.866 | 52.9 |
| 100% | 1.000 | 0.863 | 0.862 | 104.7 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe only on cuda.
