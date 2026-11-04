| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit | cuda | 0.581 | 0.616 | 0.763 | n/a | 87.4 | 1540.8 | 20.8 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.573 ± 0.003 | 0.604 ± 0.005 | 0.745 ± 0.004 | 0.724 ± 0.003 | 5.4 | 68.8 | 464.8 | 91 | 16.1x |
| student, ONNX fp32 | cpu | 0.575 | 0.607 | 0.741 | 0.725 | 3.8 | 57.8 | 554.0 | 91 | 23.3x |
| student, ONNX int8 | cpu | 0.574 | 0.606 | 0.743 | 0.721 | 1.6 | 29.0 | 1103.9 | 23 | 55.0x |

Task `clinc150`, 5500 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 15250 unlabelled training texts. Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 9, 10. The teacher scores only the 20 labels most similar to each text and always oos; the gold label was among them for 96.8% of test texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.079 | 0.574 | 0.606 | 1.6 |
| 10% | 0.373 | 0.584 | 0.612 | 10.3 |
| 20% | 0.526 | 0.586 | 0.614 | 19.1 |
| 30% | 0.672 | 0.585 | 0.612 | 27.8 |
| 50% | 0.886 | 0.585 | 0.618 | 45.3 |
| 100% | 1.000 | 0.581 | 0.616 | 88.9 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot, 4-bit on cuda.
