| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.914 | 0.917 | 0.950 | n/a | 1339.7 | 42585.8 | 0.8 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.850 ± 0.002 | 0.880 ± 0.002 | 0.913 ± 0.002 | 0.863 ± 0.003 | 4.6 | 52.6 | 608.6 | 91 | 290.5x |
| student, ONNX fp32 | cpu | 0.849 | 0.880 | 0.916 | 0.860 | 2.9 | 48.5 | 659.7 | 91 | 464.1x |
| student, ONNX int8 | cpu | 0.849 | 0.879 | 0.916 | 0.858 | 1.3 | 24.5 | 1303.7 | 23 | 1027.9x |

Task `clinc150`, 5500 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 1208 gold training examples. The student learns from the teacher's soft labels on all 15250 training texts, plus cross-entropy on 1208 gold-labelled texts (8 per class; weight 1.0). Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 10. The teacher scores only the 20 labels most similar to each text and always oos; the gold label was among them for 96.8% of test texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.078 | 0.849 | 0.879 | 1.3 |
| 10% | 0.547 | 0.892 | 0.906 | 135.3 |
| 20% | 0.829 | 0.909 | 0.916 | 269.2 |
| 30% | 0.930 | 0.912 | 0.916 | 403.2 |
| 50% | 0.967 | 0.915 | 0.918 | 671.2 |
| 100% | 1.000 | 0.914 | 0.917 | 1341.0 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit on cuda.
