| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit, probe only | cuda | 0.410 | 0.173 | 0.641 | n/a | 234.5 | 3712.4 | 8.6 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.435 ± 0.001 | 0.378 ± 0.010 | 0.683 ± 0.002 | 0.860 ± 0.003 | 11.5 | 343.9 | 93.1 | 91 | 20.4x |
| student, ONNX fp32 | cpu | 0.435 | 0.366 | 0.680 | 0.861 | 7.8 | 390.4 | 82.0 | 91 | 30.0x |
| student, ONNX int8 | cpu | 0.434 | 0.365 | 0.685 | 0.861 | 3.8 | 244.4 | 130.9 | 23 | 62.2x |
| baseline: student on 8 gold labels per class, no teacher | cpu | 0.377 ± 0.027 | 0.308 ± 0.024 | 0.540 ± 0.023 | 0.289 ± 0.008 | n/a | n/a | n/a | n/a | n/a |

Task `ap106`, 6148 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 259 gold training texts; fit on 207 of them, teacher accuracy on the other 52 is 42.3%. The student learns from the teacher's soft labels on all 22036 training texts, plus cross-entropy on 639 gold-labelled texts (8 per class; weight 1.0). The baseline is the same student trained with cross-entropy on those 639 gold labels only, without the teacher. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.079 | 0.434 | 0.365 | 3.8 |
| 10% | 0.337 | 0.431 | 0.341 | 27.2 |
| 20% | 0.438 | 0.429 | 0.336 | 50.7 |
| 30% | 0.529 | 0.429 | 0.317 | 74.1 |
| 50% | 0.723 | 0.423 | 0.237 | 121.0 |
| 100% | 1.000 | 0.410 | 0.173 | 238.3 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot, 4-bit, probe only on cuda.
