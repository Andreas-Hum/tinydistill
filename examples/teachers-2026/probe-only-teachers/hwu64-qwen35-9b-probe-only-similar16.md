| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe only | cuda | 0.887 | 0.879 | 0.962 | n/a | 100.8 | 2159.9 | 14.8 | 7530 | 1.0x |
| student, PyTorch | cpu | 0.888 ± 0.004 | 0.885 ± 0.006 | 0.966 ± 0.004 | 0.897 ± 0.006 | 3.7 | 48.4 | 661.8 | 91 | 27.5x |
| student, ONNX fp32 | cpu | 0.887 | 0.881 | 0.968 | 0.889 | 2.2 | 50.8 | 630.3 | 91 | 44.8x |
| student, ONNX int8 | cpu | 0.888 | 0.883 | 0.964 | 0.888 | 0.8 | 22.4 | 1430.7 | 23 | 134.3x |

Task `hwu64`, 1076 gold-labelled test texts. Teacher: `Qwen/Qwen3.5-9B`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 512 gold training texts; fit on 410 of them, teacher accuracy on the other 102 is 84.3%. The teacher's prompt shows each text the 16 most similar of 512 gold training examples. The student learns from the teacher's soft labels on all 8954 training texts, plus cross-entropy on 512 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.138 | 0.888 | 0.883 | 0.8 |
| 10% | 0.646 | 0.898 | 0.894 | 10.9 |
| 20% | 0.865 | 0.901 | 0.895 | 20.9 |
| 30% | 0.928 | 0.897 | 0.890 | 31.0 |
| 50% | 0.965 | 0.888 | 0.880 | 51.1 |
| 100% | 1.000 | 0.887 | 0.879 | 101.5 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe only on cuda.
