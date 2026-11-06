| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher ensemble of 2, 16 similar examples, 4-bit | cuda | 0.914 | 0.918 | 0.953 | n/a | 1923.9 | n/a | 0.5 | 15034 | 1.0x |
| teacher, 16 similar examples, 4-bit: Qwen3.5-9B | cuda | 0.841 | 0.864 | 0.931 | n/a | 482.4 | 11158.7 | 2.9 | 7530 | 4.0x |
| teacher, 16 similar examples, 4-bit: gemma-4-12B-it | cuda | 0.914 | 0.917 | 0.950 | n/a | 1441.4 | n/a | 0.7 | 7504 | 1.3x |
| student, PyTorch | cpu | 0.836 ± 0.001 | 0.873 ± 0.001 | 0.907 ± 0.001 | 0.856 ± 0.001 | 4.2 | 51.5 | 621.8 | 91 | 454.5x |
| student, ONNX fp32 | cpu | 0.837 | 0.872 | 0.906 | 0.855 | 2.8 | 49.9 | 641.1 | 91 | 692.5x |
| student, ONNX int8 | cpu | 0.839 | 0.874 | 0.909 | 0.854 | 1.1 | 24.7 | 1295.9 | 23 | 1740.1x |

Task `clinc150`, 5500 gold-labelled test texts. Teacher: `Qwen/Qwen3.5-9B,google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 1208 gold training examples. The student learns from the teacher's soft labels on all 15250 training texts, plus cross-entropy on 1208 gold-labelled texts (8 per class; weight 1.0). Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 10. The teacher scores only the 20 labels most similar to each text and always oos; the gold label was among them for 96.8% of test texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher ensemble of 2, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.072 | 0.839 | 0.874 | 1.1 |
| 10% | 0.453 | 0.882 | 0.897 | 193.5 |
| 20% | 0.704 | 0.905 | 0.913 | 385.9 |
| 30% | 0.878 | 0.911 | 0.916 | 578.3 |
| 50% | 0.958 | 0.914 | 0.918 | 963.0 |
| 100% | 1.000 | 0.914 | 0.918 | 1925.0 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher ensemble of 2, 16 similar examples, 4-bit on cuda.
