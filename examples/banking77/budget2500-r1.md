| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.851 | 0.850 | 0.947 | n/a | 175.7 | 5187.4 | 6.2 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.735 ± 0.011 | 0.705 ± 0.016 | 0.914 ± 0.005 | 0.751 ± 0.008 | 4.9 | 85.4 | 374.9 | 91 | 35.9x |
| student, ONNX fp32 | cpu | 0.725 | 0.691 | 0.912 | 0.743 | 2.9 | 79.4 | 402.8 | 91 | 60.6x |
| student, ONNX int8 | cpu | 0.717 | 0.682 | 0.909 | 0.736 | 1.1 | 36.0 | 888.5 | 23 | 167.1x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 2500 training texts, plus cross-entropy on 616 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.051 | 0.717 | 0.682 | 1.1 |
| 10% | 0.159 | 0.743 | 0.717 | 18.6 |
| 20% | 0.187 | 0.767 | 0.746 | 36.2 |
| 30% | 0.216 | 0.792 | 0.777 | 53.8 |
| 50% | 0.281 | 0.821 | 0.814 | 88.9 |
| 100% | 1.000 | 0.851 | 0.850 | 176.7 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit on cuda.
