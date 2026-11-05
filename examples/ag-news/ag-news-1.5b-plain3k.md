| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot | cuda | 0.755 | 0.756 | n/a | 30.5 | 141.0 | 227.0 | 3087 | 1.0x |
| student, PyTorch | cpu | 0.785 ± 0.002 | 0.785 ± 0.002 | 0.825 ± 0.010 | 9.4 | 399.6 | 80.1 | 91 | 3.3x |
| student, ONNX fp32 | cpu | 0.786 | 0.787 | 0.825 | 6.9 | 399.7 | 80.1 | 91 | 4.4x |
| student, ONNX int8 | cpu | 0.780 | 0.781 | 0.818 | 3.4 | 208.9 | 153.2 | 23 | 9.0x |

Task `ag-news`, 2000 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 3000 unlabelled training texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.326 | 0.780 | 0.781 | 3.4 |
| 10% | 0.468 | 0.785 | 0.786 | 6.4 |
| 20% | 0.525 | 0.785 | 0.785 | 9.5 |
| 30% | 0.582 | 0.783 | 0.783 | 12.5 |
| 50% | 0.706 | 0.771 | 0.772 | 18.6 |
| 100% | 1.000 | 0.755 | 0.756 | 33.9 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot on cuda.
