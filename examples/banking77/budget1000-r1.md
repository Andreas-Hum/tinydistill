| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.851 | 0.850 | 0.947 | n/a | 161.7 | 5087.3 | 6.3 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.480 ± 0.017 | 0.420 ± 0.021 | 0.765 ± 0.037 | 0.497 ± 0.015 | 5.3 | 87.2 | 367.2 | 91 | 30.7x |
| student, ONNX fp32 | cpu | 0.499 | 0.442 | 0.771 | 0.514 | 2.8 | 79.4 | 402.8 | 91 | 57.4x |
| student, ONNX int8 | cpu | 0.491 | 0.432 | 0.766 | 0.508 | 1.1 | 37.7 | 849.8 | 23 | 149.3x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 1000 training texts, plus cross-entropy on 590 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.025 | 0.491 | 0.432 | 1.1 |
| 10% | 0.058 | 0.544 | 0.506 | 17.3 |
| 20% | 0.071 | 0.591 | 0.564 | 33.4 |
| 30% | 0.080 | 0.627 | 0.607 | 49.6 |
| 50% | 0.096 | 0.697 | 0.682 | 82.0 |
| 100% | 1.000 | 0.851 | 0.850 | 162.8 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit on cuda.
