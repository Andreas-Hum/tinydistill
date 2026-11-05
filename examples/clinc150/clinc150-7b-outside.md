| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | In-scope accuracy | Out-of-scope recall | Out-of-scope precision | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit | cuda | 0.575 | 0.600 | 0.781 | 0.648 | 0.248 | 0.883 | n/a | 71.4 | 1301.7 | 24.6 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.567 ± 0.007 | 0.594 ± 0.004 | 0.744 ± 0.004 | 0.660 ± 0.005 | 0.151 ± 0.016 | 0.809 ± 0.036 | 0.719 ± 0.005 | 4.0 | 51.6 | 619.8 | 91 | 17.6x |
| student, ONNX fp32 | cpu | 0.560 | 0.591 | 0.739 | 0.655 | 0.133 | 0.782 | 0.715 | 2.6 | 49.1 | 652.2 | 91 | 27.1x |
| student, ONNX int8 | cpu | 0.563 | 0.590 | 0.739 | 0.655 | 0.147 | 0.790 | 0.715 | 1.0 | 22.7 | 1409.1 | 23 | 71.9x |

Task `clinc150-outside`, 5500 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 15250 unlabelled training texts. Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 10. The teacher scores only the 20 labels most similar to each text and always oos; the gold label was among them for 96.8% of test texts. In-scope accuracy is the accuracy on the test texts whose gold label is not the `outside` label (texts that fit none of the others); out-of-scope recall and precision are those of that label. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.081 | 0.563 | 0.590 | 1.0 |
| 10% | 0.382 | 0.578 | 0.599 | 8.1 |
| 20% | 0.534 | 0.587 | 0.609 | 15.3 |
| 30% | 0.671 | 0.586 | 0.609 | 22.4 |
| 50% | 0.867 | 0.580 | 0.603 | 36.7 |
| 100% | 1.000 | 0.575 | 0.600 | 72.4 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot, 4-bit on cuda.
