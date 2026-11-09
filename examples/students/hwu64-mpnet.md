| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe | cuda:0 | 0.877 | 0.876 | 0.948 | n/a | n/a | n/a | n/a | 23920 | n/a |
| student, PyTorch | cpu | 0.891 ± 0.002 | 0.891 ± 0.001 | 0.964 ± 0.006 | 0.904 ± 0.002 | 20.8 | 345.3 | 92.7 | 438 | n/a |
| student, ONNX fp32 | cpu | 0.891 | 0.892 | 0.970 | 0.903 | 14.8 | 334.9 | 95.5 | 438 | n/a |
| student, ONNX int8 | cpu | 0.864 | 0.867 | 0.952 | 0.876 | 5.3 | 113.8 | 281.3 | 110 | n/a |

Task `hwu64`, 1076 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-mpnet-base-v2`. The teacher's probabilities are averaged with those of a linear probe on its hidden states, fit on 512 gold training texts; fit on 410 of them, teacher accuracy on the other 102 is 88.2% in context, 86.3% for the probe and 88.2% combined. The teacher's prompt shows each text the 16 most similar of 512 gold training examples. The student learns from the teacher's soft labels on all 8954 training texts, plus cross-entropy on 512 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. 

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.157 | 0.864 | 0.867 | n/a |
| 10% | 0.542 | 0.892 | 0.894 | n/a |
| 20% | 0.791 | 0.888 | 0.887 | n/a |
| 30% | 0.893 | 0.887 | 0.886 | n/a |
| 50% | 0.959 | 0.878 | 0.877 | n/a |
| 100% | 1.000 | 0.877 | 0.876 | n/a |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher.
