| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.841 | 0.864 | 0.931 | n/a | 415.0 | 11758.1 | 2.7 | 7530 | 1.0x |
| student, PyTorch | cpu | 0.805 ± 0.005 | 0.848 ± 0.004 | 0.880 ± 0.007 | 0.822 ± 0.003 | 3.8 | 48.7 | 656.8 | 91 | 109.4x |
| student, ONNX fp32 | cpu | 0.799 | 0.845 | 0.872 | 0.819 | 2.2 | 47.9 | 668.0 | 91 | 184.5x |
| student, ONNX int8 | cpu | 0.799 | 0.846 | 0.873 | 0.815 | 1.0 | 23.7 | 1347.6 | 23 | 398.4x |

Task `clinc150`, 5500 gold-labelled test texts. Teacher: `Qwen/Qwen3.5-9B`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 1208 gold training examples. The student learns from the teacher's soft labels on all 15250 training texts, plus cross-entropy on 1208 gold-labelled texts (8 per class; weight 1.0). Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 9, 10. The teacher scores only the 20 labels most similar to each text and always oos; the gold label was among them for 96.8% of test texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.083 | 0.799 | 0.846 | 1.0 |
| 10% | 0.423 | 0.831 | 0.861 | 42.5 |
| 20% | 0.681 | 0.844 | 0.869 | 84.0 |
| 30% | 0.863 | 0.846 | 0.871 | 125.5 |
| 50% | 0.954 | 0.840 | 0.864 | 208.5 |
| 100% | 1.000 | 0.841 | 0.864 | 416.0 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit on cuda.
