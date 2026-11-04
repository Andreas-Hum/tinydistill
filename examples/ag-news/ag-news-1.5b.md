| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot | cuda | 0.749 | 0.752 | n/a | 32.1 | 141.8 | 225.7 | 3087 | 1.0x |
| student, PyTorch | cpu | 0.775 ± 0.009 | 0.777 ± 0.009 | 0.830 ± 0.004 | 9.2 | 374.7 | 85.4 | 91 | 3.5x |
| student, ONNX fp32 | cpu | 0.785 | 0.788 | 0.831 | 7.8 | 449.6 | 71.2 | 91 | 4.1x |
| student, ONNX int8 | cpu | 0.781 | 0.783 | 0.833 | 3.7 | 268.9 | 119.0 | 23 | 8.6x |
| upper bound: student on all gold labels | cpu | 0.896 ± 0.003 | 0.895 ± 0.002 | 0.741 ± 0.007 | n/a | n/a | n/a | n/a | n/a |

Task `ag-news`, 7600 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 10000 unlabelled training texts. The upper bound is the same student trained with cross-entropy on all 10000 gold training labels and no teacher; its speed equals the student's. Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 2, 3, 7, 8. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.269 | 0.781 | 0.783 | 3.7 |
| 10% | 0.459 | 0.781 | 0.783 | 6.9 |
| 20% | 0.517 | 0.776 | 0.779 | 10.2 |
| 30% | 0.570 | 0.772 | 0.774 | 13.4 |
| 50% | 0.678 | 0.758 | 0.760 | 19.8 |
| 100% | 1.000 | 0.749 | 0.752 | 35.9 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot on cuda.
