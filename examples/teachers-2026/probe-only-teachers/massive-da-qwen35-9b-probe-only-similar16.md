| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe only | cuda | 0.776 | 0.748 | 0.900 | n/a | 116.2 | 2353.8 | 13.6 | 7530 | 1.0x |
| student, PyTorch | cpu | 0.794 ± 0.002 | 0.764 ± 0.001 | 0.915 ± 0.001 | 0.844 ± 0.001 | 6.8 | 94.8 | 337.4 | 95 | 17.0x |
| student, ONNX fp32 | cpu | 0.796 | 0.764 | 0.916 | 0.845 | 4.7 | 93.6 | 341.8 | 96 | 24.8x |
| student, ONNX int8 | cpu | 0.791 | 0.762 | 0.915 | 0.844 | 1.8 | 40.6 | 788.3 | 24 | 64.8x |

Task `massive-da`, 2974 gold-labelled test texts. Teacher: `Qwen/Qwen3.5-9B`. Student: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. The teacher is a linear probe on its hidden states, fit on 467 gold training texts; fit on 374 of them, teacher accuracy on the other 93 is 77.4%. The teacher's prompt shows each text the 16 most similar of 476 gold training examples. The student learns from the teacher's soft labels on all 11514 training texts, plus cross-entropy on 476 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.113 | 0.791 | 0.762 | 1.8 |
| 10% | 0.509 | 0.797 | 0.768 | 13.4 |
| 20% | 0.693 | 0.795 | 0.765 | 25.0 |
| 30% | 0.859 | 0.790 | 0.762 | 36.6 |
| 50% | 0.952 | 0.782 | 0.753 | 59.9 |
| 100% | 1.000 | 0.776 | 0.748 | 118.0 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe only on cuda.
