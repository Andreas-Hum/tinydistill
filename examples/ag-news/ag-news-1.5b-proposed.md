| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot | cuda | 0.778 | 0.782 | n/a | 30.3 | 138.6 | 230.9 | 3087 | 1.0x |
| student, PyTorch | cpu | 0.803 ± 0.002 | 0.806 ± 0.002 | 0.863 ± 0.002 | 13.0 | 520.3 | 61.5 | 91 | 2.3x |
| student, ONNX fp32 | cpu | 0.801 | 0.805 | 0.860 | 10.6 | 596.0 | 53.7 | 91 | 2.9x |
| student, ONNX int8 | cpu | 0.804 | 0.807 | 0.862 | 4.2 | 309.6 | 103.4 | 23 | 7.2x |

Task `ag-news`, 7600 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 10000 unlabelled training texts. Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 7, 9. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.274 | 0.804 | 0.807 | 4.2 |
| 10% | 0.499 | 0.804 | 0.807 | 7.3 |
| 20% | 0.567 | 0.794 | 0.798 | 10.3 |
| 30% | 0.635 | 0.791 | 0.794 | 13.3 |
| 50% | 0.764 | 0.783 | 0.786 | 19.4 |
| 100% | 1.000 | 0.778 | 0.782 | 34.5 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot on cuda.
