| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, 8-shot | cuda | 0.513 | 0.473 | n/a | 95.2 | 2580.4 | 12.4 | 3087 | 1.0x |
| student, PyTorch | cpu | 0.485 ± 0.009 | 0.406 ± 0.007 | 0.706 ± 0.002 | 4.5 | 85.2 | 375.5 | 91 | 21.3x |
| student, ONNX fp32 | cpu | 0.495 | 0.415 | 0.709 | 2.7 | 79.5 | 402.5 | 91 | 35.8x |
| student, ONNX int8 | cpu | 0.488 | 0.407 | 0.702 | 1.1 | 36.4 | 878.1 | 23 | 82.8x |
| baseline: student on 8 gold labels per class, no teacher | cpu | 0.490 ± 0.014 | 0.449 ± 0.018 | 0.363 ± 0.028 | n/a | n/a | n/a | n/a | n/a |
| upper bound: student on all gold labels | cpu | 0.914 ± 0.005 | 0.914 ± 0.005 | 0.513 ± 0.002 | n/a | n/a | n/a | n/a | n/a |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows 8 gold training examples per class (616 texts). The student learns from the teacher's soft labels on all 10003 training texts, plus cross-entropy on 8 gold labels per class (616 texts, weight 1.0). The baseline is the same student trained with cross-entropy on those 616 gold labels only, without the teacher. The upper bound is the same student trained with cross-entropy on all 10003 gold training labels and no teacher; its speed equals the student's. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 8-shot on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.055 | 0.488 | 0.407 | 1.1 |
| 10% | 0.181 | 0.492 | 0.422 | 10.7 |
| 20% | 0.246 | 0.501 | 0.440 | 20.2 |
| 30% | 0.293 | 0.509 | 0.451 | 29.7 |
| 50% | 0.403 | 0.506 | 0.455 | 48.7 |
| 100% | 1.000 | 0.513 | 0.473 | 96.3 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 8-shot on cuda.
