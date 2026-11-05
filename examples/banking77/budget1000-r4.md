| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.851 | 0.850 | 0.947 | n/a | 170.6 | 4988.4 | 6.4 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.383 ± 0.016 | 0.319 ± 0.018 | 0.652 ± 0.003 | 0.390 ± 0.018 | 5.1 | 83.2 | 384.5 | 91 | 33.7x |
| student, ONNX fp32 | cpu | 0.400 | 0.339 | 0.655 | 0.407 | 3.1 | 80.1 | 399.3 | 91 | 54.3x |
| student, ONNX int8 | cpu | 0.398 | 0.333 | 0.656 | 0.407 | 1.4 | 38.7 | 825.8 | 23 | 118.9x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 1000 training texts, plus cross-entropy on 512 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.027 | 0.398 | 0.333 | 1.4 |
| 10% | 0.042 | 0.452 | 0.415 | 18.5 |
| 20% | 0.049 | 0.506 | 0.482 | 35.6 |
| 30% | 0.054 | 0.559 | 0.545 | 52.6 |
| 50% | 0.066 | 0.661 | 0.657 | 86.7 |
| 100% | 1.000 | 0.851 | 0.850 | 172.1 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit on cuda.
