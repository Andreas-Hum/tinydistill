| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe | cuda | 0.808 | 0.780 | 0.908 | n/a | 2566.5 | n/a | 0.4 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.768 ± 0.004 | 0.721 ± 0.003 | 0.894 ± 0.004 | 0.797 ± 0.003 | 4.8 | 76.3 | 419.2 | 91 | 537.5x |
| student, ONNX fp32 | cpu | 0.769 | 0.718 | 0.899 | 0.794 | 3.0 | 69.0 | 463.5 | 91 | 858.3x |
| student, ONNX int8 | cpu | 0.763 | 0.711 | 0.898 | 0.789 | 1.1 | 31.9 | 1003.0 | 23 | 2248.4x |

Task `massive-da`, 2974 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's probabilities are averaged with those of a linear probe on its hidden states, fit on 467 gold training texts; fit on 374 of them, teacher accuracy on the other 93 is 81.7% in context, 76.3% for the probe and 82.8% combined. The teacher's prompt shows each text the 16 most similar of 476 gold training examples. The student learns from the teacher's soft labels on all 11514 training texts, plus cross-entropy on 476 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.129 | 0.763 | 0.711 | 1.1 |
| 10% | 0.412 | 0.789 | 0.743 | 257.5 |
| 20% | 0.563 | 0.806 | 0.773 | 514.6 |
| 30% | 0.711 | 0.813 | 0.781 | 770.9 |
| 50% | 0.872 | 0.809 | 0.781 | 1284.4 |
| 100% | 1.000 | 0.808 | 0.780 | 2567.7 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe on cuda.
