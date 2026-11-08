| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit, probe only | cuda | 0.842 | 0.842 | 0.936 | n/a | 156.3 | 1404.7 | 22.8 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.865 ± 0.001 | 0.865 ± 0.004 | 0.953 ± 0.004 | 0.880 ± 0.003 | 5.8 | 65.6 | 487.9 | 91 | 27.1x |
| student, ONNX fp32 | cpu | 0.863 | 0.860 | 0.948 | 0.877 | 3.4 | 62.1 | 515.1 | 91 | 45.5x |
| student, ONNX int8 | cpu | 0.864 | 0.864 | 0.947 | 0.882 | 1.1 | 30.9 | 1036.9 | 23 | 144.5x |
| baseline: student on 8 gold labels per class, no teacher | cpu | 0.639 ± 0.004 | 0.576 ± 0.012 | 0.871 ± 0.001 | 0.604 ± 0.007 | n/a | n/a | n/a | n/a | n/a |

Task `hwu64`, 1076 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 512 gold training texts; fit on 410 of them, teacher accuracy on the other 102 is 87.3%. The student learns from the teacher's soft labels on all 8954 training texts, plus cross-entropy on 512 gold-labelled texts (8 per class; weight 1.0). The baseline is the same student trained with cross-entropy on those 512 gold labels only, without the teacher. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.163 | 0.864 | 0.864 | 1.1 |
| 10% | 0.612 | 0.871 | 0.869 | 16.8 |
| 20% | 0.823 | 0.862 | 0.859 | 32.3 |
| 30% | 0.903 | 0.852 | 0.851 | 48.0 |
| 50% | 0.953 | 0.845 | 0.844 | 79.2 |
| 100% | 1.000 | 0.842 | 0.842 | 157.4 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot, 4-bit, probe only on cuda.
