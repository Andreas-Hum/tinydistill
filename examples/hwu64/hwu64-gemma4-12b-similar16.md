| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe | cuda | 0.877 | 0.876 | 0.948 | n/a | 2196.4 | n/a | 0.5 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.894 ± 0.005 | 0.893 ± 0.004 | 0.964 ± 0.001 | 0.893 ± 0.003 | 3.7 | 48.1 | 665.0 | 91 | 593.2x |
| student, ONNX fp32 | cpu | 0.898 | 0.897 | 0.965 | 0.893 | 2.2 | 45.7 | 699.7 | 91 | 1013.4x |
| student, ONNX int8 | cpu | 0.892 | 0.891 | 0.958 | 0.890 | 0.8 | 21.7 | 1475.5 | 23 | 2927.1x |

Task `hwu64`, 1076 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's probabilities are averaged with those of a linear probe on its hidden states, fit on 512 gold training texts; fit on 410 of them, teacher accuracy on the other 102 is 88.2% in context, 86.3% for the probe and 88.2% combined. The teacher's prompt shows each text the 16 most similar of 512 gold training examples. The student learns from the teacher's soft labels on all 8954 training texts, plus cross-entropy on 512 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.193 | 0.892 | 0.891 | 0.8 |
| 10% | 0.678 | 0.898 | 0.896 | 221.2 |
| 20% | 0.869 | 0.887 | 0.885 | 439.6 |
| 30% | 0.933 | 0.879 | 0.878 | 660.1 |
| 50% | 0.962 | 0.880 | 0.878 | 1098.9 |
| 100% | 1.000 | 0.877 | 0.876 | 2197.1 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe on cuda.
