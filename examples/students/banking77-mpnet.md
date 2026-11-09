| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe | cuda:0 | 0.859 | 0.859 | 0.946 | n/a | n/a | n/a | n/a | n/a | n/a |
| student, PyTorch | cpu | 0.882 ± 0.001 | 0.882 ± 0.001 | 0.961 ± 0.004 | 0.898 ± 0.003 | 24.2 | 595.9 | 53.7 | 438 | n/a |
| student, ONNX fp32 | cpu | 0.882 | 0.882 | 0.965 | 0.900 | 19.1 | 538.6 | 59.4 | 438 | n/a |
| student, ONNX int8 | cpu | 0.868 | 0.868 | 0.956 | 0.882 | 7.4 | 198.6 | 161.2 | 110 | n/a |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-mpnet-base-v2`. The teacher's probabilities are averaged with those of a linear probe on its hidden states, fit on 616 gold training texts; fit on 493 of them, teacher accuracy on the other 123 is 78.0% in context, 75.6% for the probe and 78.9% combined. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 10003 training texts, plus cross-entropy on 616 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. 

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.153 | 0.868 | 0.868 | n/a |
| 10% | 0.536 | 0.879 | 0.878 | n/a |
| 20% | 0.743 | 0.871 | 0.871 | n/a |
| 30% | 0.858 | 0.867 | 0.866 | n/a |
| 50% | 0.946 | 0.861 | 0.861 | n/a |
| 100% | 1.000 | 0.859 | 0.859 | n/a |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher.
