| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.821 | 0.770 | n/a | 171.9 | 4907.7 | 6.5 | 7530 | 1.0x |
| student, PyTorch | cpu | 0.805 ± 0.005 | 0.752 ± 0.006 | 0.836 ± 0.004 | 4.2 | 137.2 | 233.3 | 91 | 41.1x |
| student, ONNX fp32 | cpu | 0.810 | 0.757 | 0.835 | 2.9 | 145.3 | 220.2 | 91 | 60.3x |
| student, ONNX int8 | cpu | 0.816 | 0.758 | 0.833 | 1.2 | 66.9 | 478.1 | 23 | 143.6x |

Task `financial-tweets`, 2388 gold-labelled test texts. Teacher: `Qwen/Qwen3.5-9B`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 24 gold training examples. The student learns from the teacher's soft labels on all 9543 training texts, plus cross-entropy on 24 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.345 | 0.816 | 0.758 | 1.2 |
| 10% | 0.558 | 0.832 | 0.778 | 18.4 |
| 20% | 0.683 | 0.835 | 0.782 | 35.6 |
| 30% | 0.780 | 0.828 | 0.775 | 52.7 |
| 50% | 0.925 | 0.824 | 0.771 | 87.1 |
| 100% | 1.000 | 0.821 | 0.770 | 173.1 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit on cuda.
