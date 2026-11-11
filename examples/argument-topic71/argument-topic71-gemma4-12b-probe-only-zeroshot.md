| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit, probe only | cuda | 0.977 | 0.975 | 0.987 | n/a | 151.1 | 2509.2 | 12.8 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.960 ± 0.000 | 0.958 ± 0.000 | 0.979 ± 0.001 | 0.957 ± 0.001 | 7.6 | 178.0 | 179.8 | 91 | 20.0x |
| student, ONNX fp32 | cpu | 0.959 | 0.958 | 0.978 | 0.955 | 4.7 | 145.3 | 220.2 | 91 | 31.8x |
| student, ONNX int8 | cpu | 0.960 | 0.959 | 0.979 | 0.956 | 2.4 | 69.1 | 463.3 | 23 | 63.3x |
| baseline: student on 8 gold labels per class, no teacher | cpu | 0.941 ± 0.003 | 0.939 ± 0.002 | 0.965 ± 0.004 | 0.940 ± 0.002 | n/a | n/a | n/a | n/a | n/a |

Task `argument-topic71`, 1898 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 568 gold training texts; fit on 454 of them, teacher accuracy on the other 114 is 93.0%. The student learns from the teacher's soft labels on all 6640 training texts, plus cross-entropy on 568 gold-labelled texts (8 per class; weight 1.0). The baseline is the same student trained with cross-entropy on those 568 gold labels only, without the teacher. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.122 | 0.960 | 0.959 | 2.4 |
| 10% | 0.961 | 0.977 | 0.975 | 17.5 |
| 20% | 0.980 | 0.977 | 0.975 | 32.6 |
| 30% | 0.982 | 0.977 | 0.975 | 47.7 |
| 50% | 0.986 | 0.977 | 0.975 | 77.9 |
| 100% | 1.000 | 0.977 | 0.975 | 153.5 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot, 4-bit, probe only on cuda.
