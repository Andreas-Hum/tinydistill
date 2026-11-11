| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit, probe only | cuda | 0.896 | 0.862 | 0.946 | n/a | 139.7 | 1713.0 | 18.7 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.844 ± 0.008 | 0.766 ± 0.018 | 0.931 ± 0.007 | 0.837 ± 0.002 | 6.0 | 122.3 | 261.6 | 91 | 23.4x |
| student, ONNX fp32 | cpu | 0.835 | 0.766 | 0.933 | 0.835 | 3.1 | 105.1 | 304.4 | 91 | 44.4x |
| student, ONNX int8 | cpu | 0.831 | 0.758 | 0.935 | 0.840 | 1.7 | 50.2 | 637.6 | 23 | 84.1x |
| baseline: student on 8 gold labels per class, no teacher | cpu | 0.784 ± 0.001 | 0.701 ± 0.020 | 0.894 ± 0.010 | 0.782 ± 0.001 | n/a | n/a | n/a | n/a | n/a |

Task `claim-stance55`, 480 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 422 gold training texts; fit on 338 of them, teacher accuracy on the other 84 is 89.3%. The student learns from the teacher's soft labels on all 1675 training texts, plus cross-entropy on 424 gold-labelled texts (8 per class; weight 1.0). The baseline is the same student trained with cross-entropy on those 424 gold labels only, without the teacher. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.055 | 0.831 | 0.758 | 1.7 |
| 10% | 0.164 | 0.869 | 0.831 | 15.6 |
| 20% | 0.249 | 0.894 | 0.856 | 29.6 |
| 30% | 0.309 | 0.896 | 0.857 | 43.6 |
| 50% | 0.537 | 0.902 | 0.866 | 71.5 |
| 100% | 1.000 | 0.896 | 0.862 | 141.4 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot, 4-bit, probe only on cuda.
