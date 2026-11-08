| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe only | cuda | 0.925 | 0.945 | 0.951 | n/a | 384.2 | 4514.1 | 7.1 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.852 ± 0.002 | 0.896 ± 0.004 | 0.901 ± 0.006 | 0.859 ± 0.002 | 7.1 | 86.3 | 370.9 | 91 | 54.2x |
| student, ONNX fp32 | cpu | 0.853 | 0.900 | 0.901 | 0.860 | 4.4 | 78.1 | 409.8 | 91 | 88.1x |
| student, ONNX int8 | cpu | 0.851 | 0.897 | 0.900 | 0.860 | 1.6 | 34.1 | 939.5 | 23 | 242.9x |

Task `clinc150`, 5500 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 1208 gold training texts; fit on 966 of them, teacher accuracy on the other 242 is 98.3%. The teacher's prompt shows each text the 16 most similar of 1208 gold training examples. The student learns from the teacher's soft labels on all 15250 training texts, plus cross-entropy on 1208 gold-labelled texts (8 per class; weight 1.0). Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 9, 10. The teacher scores only the 20 labels most similar to each text and always oos; the gold label was among them for 100.0% of test texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.081 | 0.851 | 0.897 | 1.6 |
| 10% | 0.526 | 0.897 | 0.926 | 40.0 |
| 20% | 0.853 | 0.918 | 0.941 | 78.4 |
| 30% | 0.943 | 0.922 | 0.944 | 116.9 |
| 50% | 0.971 | 0.925 | 0.945 | 193.7 |
| 100% | 1.000 | 0.925 | 0.945 | 385.8 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe only on cuda.
