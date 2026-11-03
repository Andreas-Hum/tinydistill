| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit | cuda:0 | 0.832 | 0.788 | n/a | n/a | n/a | n/a | n/a | n/a |
| student, PyTorch | cpu | 0.596 ± 0.037 | 0.571 ± 0.027 | 0.615 ± 0.045 | 7.1 | 201.9 | 158.5 | 91 | n/a |
| student, ONNX fp32 | cpu | 0.603 | 0.577 | 0.630 | 4.5 | 205.5 | 155.7 | 91 | n/a |
| student, ONNX int8 | cpu | 0.589 | 0.565 | 0.609 | 1.6 | 110.4 | 289.8 | 23 | n/a |

Task `financial-tweets`, 2388 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 1498 unlabelled training texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. 

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.347 | 0.589 | 0.565 | n/a |
| 10% | 0.647 | 0.624 | 0.596 | n/a |
| 20% | 0.786 | 0.662 | 0.629 | n/a |
| 30% | 0.879 | 0.695 | 0.661 | n/a |
| 50% | 0.954 | 0.749 | 0.709 | n/a |
| 100% | 1.000 | 0.832 | 0.788 | n/a |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher.
