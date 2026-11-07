| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.781 | 0.756 | 0.896 | n/a | 826.2 | 25900.4 | 1.2 | 7530 | 1.0x |
| student, PyTorch | cpu | 0.756 ± 0.002 | 0.700 ± 0.004 | 0.879 ± 0.004 | 0.782 ± 0.002 | 4.3 | 67.8 | 472.0 | 91 | 191.7x |
| student, ONNX fp32 | cpu | 0.753 | 0.695 | 0.883 | 0.784 | 2.3 | 63.8 | 501.4 | 91 | 361.6x |
| student, ONNX int8 | cpu | 0.743 | 0.684 | 0.880 | 0.777 | 0.9 | 30.4 | 1052.7 | 23 | 943.6x |

Task `massive-da`, 2974 gold-labelled test texts. Teacher: `Qwen/Qwen3.5-9B`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 476 gold training examples. The student learns from the teacher's soft labels on all 11514 training texts, plus cross-entropy on 476 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.094 | 0.743 | 0.684 | 0.9 |
| 10% | 0.402 | 0.767 | 0.740 | 83.4 |
| 20% | 0.528 | 0.779 | 0.757 | 166.2 |
| 30% | 0.679 | 0.790 | 0.766 | 248.7 |
| 50% | 0.865 | 0.782 | 0.756 | 414.0 |
| 100% | 1.000 | 0.781 | 0.756 | 827.0 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit on cuda.
