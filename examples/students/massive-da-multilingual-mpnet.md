| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe | cuda:0 | 0.808 | 0.780 | 0.908 | n/a | n/a | n/a | n/a | 23920 | n/a |
| student, PyTorch | cpu | 0.818 ± 0.004 | 0.793 ± 0.004 | 0.924 ± 0.003 | 0.861 ± 0.003 | 20.2 | 341.0 | 93.9 | 362 | n/a |
| student, ONNX fp32 | cpu | 0.822 | 0.792 | 0.925 | 0.863 | 15.0 | 325.8 | 98.2 | 362 | n/a |
| student, ONNX int8 | cpu | 0.821 | 0.791 | 0.924 | 0.864 | 5.6 | 119.5 | 267.7 | 91 | n/a |

Task `massive-da`, 2974 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`. The teacher's probabilities are averaged with those of a linear probe on its hidden states, fit on 467 gold training texts; fit on 374 of them, teacher accuracy on the other 93 is 81.7% in context, 76.3% for the probe and 82.8% combined. The teacher's prompt shows each text the 16 most similar of 476 gold training examples. The student learns from the teacher's soft labels on all 11514 training texts, plus cross-entropy on 476 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. 

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.155 | 0.821 | 0.791 | n/a |
| 10% | 0.571 | 0.828 | 0.800 | n/a |
| 20% | 0.767 | 0.822 | 0.795 | n/a |
| 30% | 0.885 | 0.817 | 0.787 | n/a |
| 50% | 0.968 | 0.811 | 0.784 | n/a |
| 100% | 1.000 | 0.808 | 0.780 | n/a |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher.
