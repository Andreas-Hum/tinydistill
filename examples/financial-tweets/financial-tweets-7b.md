| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit | cuda | 0.832 | 0.788 | n/a | 52.3 | 402.7 | 79.5 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.825 ± 0.007 | 0.768 ± 0.009 | 0.870 ± 0.005 | 5.6 | 168.3 | 190.1 | 91 | 9.3x |
| student, ONNX fp32 | cpu | 0.827 | 0.770 | 0.874 | 4.0 | 185.5 | 172.5 | 91 | 13.0x |
| student, ONNX int8 | cpu | 0.820 | 0.762 | 0.869 | 2.5 | 105.7 | 302.6 | 23 | 20.9x |
| upper bound: student on all gold labels | cpu | 0.874 ± 0.005 | 0.832 ± 0.007 | 0.811 ± 0.003 | n/a | n/a | n/a | n/a | n/a |

Task `financial-tweets`, 2388 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 9543 unlabelled training texts. The upper bound is the same student trained with cross-entropy on all 9543 gold training labels and no teacher; its speed equals the student's. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.447 | 0.820 | 0.762 | 2.5 |
| 10% | 0.836 | 0.830 | 0.775 | 7.7 |
| 20% | 0.940 | 0.835 | 0.785 | 13.0 |
| 30% | 0.977 | 0.838 | 0.790 | 18.2 |
| 50% | 0.993 | 0.835 | 0.790 | 28.7 |
| 100% | 1.000 | 0.832 | 0.788 | 54.8 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot, 4-bit on cuda.
