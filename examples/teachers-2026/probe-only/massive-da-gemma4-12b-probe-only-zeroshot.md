| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit, probe only | cuda | 0.760 | 0.728 | 0.882 | n/a | 123.8 | 1126.4 | 28.4 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.779 ± 0.002 | 0.744 ± 0.003 | 0.899 ± 0.004 | 0.820 ± 0.003 | 8.3 | 111.3 | 287.5 | 95 | 14.9x |
| student, ONNX fp32 | cpu | 0.778 | 0.743 | 0.895 | 0.821 | 5.1 | 100.7 | 317.7 | 96 | 24.1x |
| student, ONNX int8 | cpu | 0.778 | 0.744 | 0.896 | 0.824 | 2.4 | 45.3 | 706.8 | 24 | 52.0x |

Task `massive-da`, 2974 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. The teacher is a linear probe on its hidden states, fit on 467 gold training texts; fit on 374 of them, teacher accuracy on the other 93 is 76.3%. The student learns from the teacher's soft labels on all 11514 training texts, plus cross-entropy on 476 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.134 | 0.778 | 0.744 | 2.4 |
| 10% | 0.498 | 0.785 | 0.751 | 14.7 |
| 20% | 0.692 | 0.780 | 0.747 | 27.1 |
| 30% | 0.834 | 0.776 | 0.744 | 39.5 |
| 50% | 0.947 | 0.769 | 0.737 | 64.3 |
| 100% | 1.000 | 0.760 | 0.728 | 126.2 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot, 4-bit, probe only on cuda.
