| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit | cuda:0 | 0.832 | 0.788 | n/a | n/a | n/a | n/a | n/a | n/a |
| student, PyTorch | cpu | 0.825 ± 0.004 | 0.765 ± 0.004 | 0.871 ± 0.003 | 4.9 | 150.4 | 212.8 | 91 | n/a |
| student, ONNX fp32 | cpu | 0.825 | 0.762 | 0.870 | 4.0 | 148.2 | 215.9 | 91 | n/a |
| student, ONNX int8 | cpu | 0.820 | 0.759 | 0.871 | 1.4 | 74.8 | 427.8 | 23 | n/a |

Task `financial-tweets`, 2388 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 9543 unlabelled training texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. 

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.378 | 0.820 | 0.759 | n/a |
| 10% | 0.788 | 0.835 | 0.780 | n/a |
| 20% | 0.953 | 0.838 | 0.789 | n/a |
| 30% | 0.988 | 0.836 | 0.788 | n/a |
| 50% | 0.999 | 0.834 | 0.788 | n/a |
| 100% | 1.000 | 0.832 | 0.788 | n/a |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher.
