| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot | cuda | 0.304 | 0.207 | n/a | 28.8 | 182.8 | 175.1 | 3087 | 1.0x |
| student, PyTorch | cpu | 0.283 ± 0.006 | 0.168 ± 0.011 | 0.945 ± 0.003 | 9.9 | 424.8 | 75.3 | 91 | 2.9x |
| student, ONNX fp32 | cpu | 0.289 | 0.180 | 0.942 | 7.1 | 461.1 | 69.4 | 91 | 4.1x |
| student, ONNX int8 | cpu | 0.290 | 0.181 | 0.943 | 3.5 | 252.8 | 126.6 | 23 | 8.2x |
| upper bound: student on all gold labels | cpu | 0.895 ± 0.004 | 0.895 ± 0.003 | 0.288 ± 0.012 | n/a | n/a | n/a | n/a | n/a |

Task `ag-news`, 7600 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 10000 unlabelled training texts. The upper bound is the same student trained with cross-entropy on all 10000 gold training labels and no teacher; its speed equals the student's. Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 2, 3, 8. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.404 | 0.290 | 0.181 | 3.5 |
| 10% | 0.752 | 0.292 | 0.186 | 6.4 |
| 20% | 0.847 | 0.297 | 0.195 | 9.3 |
| 30% | 0.893 | 0.300 | 0.200 | 12.1 |
| 50% | 0.938 | 0.302 | 0.204 | 17.9 |
| 100% | 1.000 | 0.304 | 0.207 | 32.3 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot on cuda.
