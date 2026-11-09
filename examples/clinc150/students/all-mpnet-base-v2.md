| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe | cuda | 0.941 | 0.950 | 0.971 | n/a | n/a | n/a | n/a | 23920 | n/a |
| student, PyTorch | cpu | 0.894 ± 0.005 | 0.921 ± 0.003 | 0.950 ± 0.002 | 0.905 ± 0.003 | 21.9 | 375.8 | 85.2 | 438 | n/a |
| student, ONNX fp32 | cpu | 0.890 | 0.920 | 0.948 | 0.905 | 18.7 | 342.6 | 93.4 | 439 | n/a |
| student, ONNX int8 | cpu | 0.902 | 0.923 | 0.962 | 0.914 | 7.2 | 131.2 | 243.9 | 110 | n/a |

Task `clinc150`, 5500 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-mpnet-base-v2`. The teacher's probabilities are averaged with those of a linear probe on its hidden states, fit on 1208 gold training texts; fit on 966 of them, teacher accuracy on the other 242 is 94.6% in context, 98.3% for the probe and 98.8% combined. The teacher's prompt shows each text the 16 most similar of 1208 gold training examples. The student learns from the teacher's soft labels on all 15250 training texts, plus cross-entropy on 1208 gold-labelled texts (8 per class; weight 1.0). Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 9, 10. The teacher scores only the 20 labels most similar to each text and always oos; the gold label was among them for 100.0% of test texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. 

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.118 | 0.902 | 0.923 | n/a |
| 10% | 0.536 | 0.930 | 0.943 | n/a |
| 20% | 0.767 | 0.938 | 0.949 | n/a |
| 30% | 0.894 | 0.941 | 0.950 | n/a |
| 50% | 0.970 | 0.941 | 0.950 | n/a |
| 100% | 1.000 | 0.941 | 0.950 | n/a |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher.
