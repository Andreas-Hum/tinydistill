| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit | cuda | 0.603 | 0.576 | 0.808 | n/a | 104.1 | 2986.4 | 10.7 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.594 ± 0.007 | 0.536 ± 0.008 | 0.791 ± 0.004 | 0.725 ± 0.003 | 3.8 | 65.9 | 485.5 | 91 | 27.2x |
| student, ONNX fp32 | cpu | 0.588 | 0.531 | 0.787 | 0.722 | 2.3 | 62.6 | 511.3 | 91 | 45.2x |
| student, ONNX int8 | cpu | 0.583 | 0.519 | 0.790 | 0.722 | 0.8 | 29.6 | 1082.0 | 23 | 123.3x |

Task `massive-da`, 2974 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 11514 unlabelled training texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.124 | 0.583 | 0.519 | 0.8 |
| 10% | 0.397 | 0.593 | 0.538 | 11.2 |
| 20% | 0.538 | 0.605 | 0.561 | 21.7 |
| 30% | 0.674 | 0.607 | 0.574 | 32.1 |
| 50% | 0.860 | 0.607 | 0.577 | 52.9 |
| 100% | 1.000 | 0.603 | 0.576 | 105.0 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot, 4-bit on cuda.
