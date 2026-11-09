| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit, probe only | cuda | 0.722 | 0.545 | 0.880 | n/a | 117.3 | 999.4 | 32.0 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.765 ± 0.004 | 0.632 ± 0.017 | 0.919 ± 0.002 | 0.780 ± 0.009 | 3.9 | 50.7 | 631.2 | 91 | 30.1x |
| student, ONNX fp32 | cpu | 0.766 | 0.612 | 0.920 | 0.774 | 2.2 | 48.9 | 654.2 | 91 | 52.4x |
| student, ONNX int8 | cpu | 0.752 | 0.601 | 0.922 | 0.780 | 0.8 | 24.0 | 1335.0 | 23 | 146.7x |
| baseline: student on 8 gold labels per class, no teacher | cpu | 0.477 ± 0.129 | 0.383 ± 0.067 | 0.699 ± 0.164 | 0.501 ± 0.089 | n/a | n/a | n/a | n/a | n/a |

Task `trec50`, 500 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 381 gold training texts; fit on 305 of them, teacher accuracy on the other 76 is 63.2%. The student learns from the teacher's soft labels on all 5452 training texts, plus cross-entropy on 390 gold-labelled texts (8 per class; weight 1.0). The baseline is the same student trained with cross-entropy on those 390 gold labels only, without the teacher. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.130 | 0.752 | 0.601 | 0.8 |
| 10% | 0.320 | 0.758 | 0.581 | 12.5 |
| 20% | 0.432 | 0.756 | 0.583 | 24.3 |
| 30% | 0.571 | 0.766 | 0.577 | 36.0 |
| 50% | 0.739 | 0.736 | 0.548 | 59.4 |
| 100% | 1.000 | 0.722 | 0.545 | 118.1 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot, 4-bit, probe only on cuda.
