| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe only | cuda | 0.400 | 0.193 | 0.618 | n/a | 313.1 | 7326.9 | 4.4 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.433 ± 0.002 | 0.383 ± 0.008 | 0.661 ± 0.001 | 0.836 ± 0.007 | 8.2 | 289.4 | 110.6 | 91 | 38.3x |
| student, ONNX fp32 | cpu | 0.434 | 0.378 | 0.662 | 0.843 | 6.3 | 310.1 | 103.2 | 91 | 49.9x |
| student, ONNX int8 | cpu | 0.434 | 0.380 | 0.663 | 0.838 | 2.9 | 161.5 | 198.1 | 23 | 109.9x |

Task `ap106`, 6148 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 259 gold training texts; fit on 207 of them, teacher accuracy on the other 52 is 38.5%. The teacher's prompt shows each text the 16 most similar of 639 gold training examples. The student learns from the teacher's soft labels on all 22036 training texts, plus cross-entropy on 639 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.089 | 0.434 | 0.380 | 2.9 |
| 10% | 0.277 | 0.432 | 0.366 | 34.2 |
| 20% | 0.367 | 0.426 | 0.348 | 65.5 |
| 30% | 0.448 | 0.423 | 0.343 | 96.8 |
| 50% | 0.627 | 0.414 | 0.284 | 159.4 |
| 100% | 1.000 | 0.400 | 0.193 | 315.9 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe only on cuda.
