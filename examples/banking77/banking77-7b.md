| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit | cuda | 0.629 | 0.605 | n/a | 165.9 | 4919.6 | 6.5 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.618 ± 0.003 | 0.579 ± 0.005 | 0.817 ± 0.007 | 4.7 | 87.6 | 365.3 | 91 | 35.4x |
| student, ONNX fp32 | cpu | 0.621 | 0.585 | 0.825 | 3.1 | 85.3 | 375.1 | 91 | 53.1x |
| student, ONNX int8 | cpu | 0.621 | 0.586 | 0.820 | 1.4 | 39.2 | 816.2 | 23 | 121.2x |
| upper bound: student on all gold labels | cpu | 0.914 ± 0.005 | 0.914 ± 0.005 | 0.631 ± 0.001 | n/a | n/a | n/a | n/a | n/a |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 10003 unlabelled training texts. The upper bound is the same student trained with cross-entropy on all 10003 gold training labels and no teacher; its speed equals the student's. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.123 | 0.621 | 0.586 | 1.4 |
| 10% | 0.392 | 0.628 | 0.595 | 18.0 |
| 20% | 0.512 | 0.632 | 0.602 | 34.6 |
| 30% | 0.614 | 0.632 | 0.602 | 51.1 |
| 50% | 0.757 | 0.631 | 0.604 | 84.3 |
| 100% | 1.000 | 0.629 | 0.605 | 167.3 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot, 4-bit on cuda.
