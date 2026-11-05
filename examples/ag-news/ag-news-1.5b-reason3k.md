| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, reasoning | cuda | 0.778 | 0.771 | n/a | 445.4 | 751.3 | 42.6 | 3087 | 1.0x |
| student, PyTorch | cpu | 0.785 ± 0.004 | 0.777 ± 0.004 | 0.914 ± 0.003 | 8.2 | 365.9 | 87.5 | 91 | 54.4x |
| student, ONNX fp32 | cpu | 0.783 | 0.775 | 0.911 | 6.7 | 404.9 | 79.0 | 91 | 66.2x |
| student, ONNX int8 | cpu | 0.783 | 0.774 | 0.914 | 3.5 | 208.3 | 153.6 | 23 | 128.7x |

Task `ag-news`, 2000 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 3000 unlabelled training texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, reasoning on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.302 | 0.783 | 0.774 | 3.5 |
| 10% | 0.645 | 0.785 | 0.777 | 48.0 |
| 20% | 0.780 | 0.784 | 0.776 | 92.5 |
| 30% | 0.876 | 0.783 | 0.776 | 137.1 |
| 50% | 0.966 | 0.778 | 0.771 | 226.2 |
| 100% | 1.000 | 0.778 | 0.771 | 448.9 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot, reasoning on cuda.
