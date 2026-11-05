| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.851 | 0.850 | 0.947 | n/a | n/a | n/a | n/a | n/a | n/a |
| student, PyTorch | cpu | 0.510 ± 0.023 | 0.437 ± 0.030 | 0.787 ± 0.038 | 0.527 ± 0.023 | 7.1 | 119.7 | 267.2 | 91 | n/a |
| student, ONNX fp32 | cpu | 0.534 | 0.470 | 0.817 | 0.550 | 4.0 | 130.7 | 244.8 | 91 | n/a |
| student, ONNX int8 | cpu | 0.525 | 0.462 | 0.808 | 0.542 | 1.8 | 61.6 | 519.8 | 23 | n/a |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 1000 training texts, plus cross-entropy on 590 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. 

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.034 | 0.525 | 0.462 | n/a |
| 10% | 0.071 | 0.572 | 0.525 | n/a |
| 20% | 0.081 | 0.624 | 0.585 | n/a |
| 30% | 0.091 | 0.673 | 0.641 | n/a |
| 50% | 0.111 | 0.747 | 0.730 | n/a |
| 100% | 1.000 | 0.851 | 0.850 | n/a |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher.
