| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe only | cuda | 0.899 | 0.924 | 0.963 | n/a | 149.0 | 2151.1 | 14.9 | 7530 | 1.0x |
| student, PyTorch | cpu | 0.841 ± 0.004 | 0.889 ± 0.003 | 0.894 ± 0.003 | 0.856 ± 0.003 | 4.0 | 50.9 | 629.0 | 91 | 36.9x |
| student, ONNX fp32 | cpu | 0.839 | 0.889 | 0.891 | 0.858 | 2.4 | 50.8 | 630.4 | 91 | 61.4x |
| student, ONNX int8 | cpu | 0.837 | 0.887 | 0.890 | 0.856 | 0.9 | 23.3 | 1375.8 | 23 | 157.6x |

Task `clinc150`, 5500 gold-labelled test texts. Teacher: `Qwen/Qwen3.5-9B`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 1208 gold training texts; fit on 966 of them, teacher accuracy on the other 242 is 95.5%. The teacher's prompt shows each text the 16 most similar of 1208 gold training examples. The student learns from the teacher's soft labels on all 15250 training texts, plus cross-entropy on 1208 gold-labelled texts (8 per class; weight 1.0). Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 10. The teacher scores only the 20 labels most similar to each text and always oos; the gold label was among them for 100.0% of test texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.082 | 0.837 | 0.887 | 0.9 |
| 10% | 0.504 | 0.881 | 0.914 | 15.8 |
| 20% | 0.818 | 0.896 | 0.924 | 30.7 |
| 30% | 0.938 | 0.898 | 0.924 | 45.6 |
| 50% | 0.971 | 0.899 | 0.924 | 75.4 |
| 100% | 1.000 | 0.899 | 0.924 | 149.9 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe only on cuda.
