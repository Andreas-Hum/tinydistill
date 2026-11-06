| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.870 | 0.869 | 0.939 | n/a | 2643.8 | n/a | 0.4 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.870 ± 0.005 | 0.869 ± 0.004 | 0.947 ± 0.003 | 0.913 ± 0.001 | 4.3 | 78.4 | 408.1 | 91 | 621.3x |
| student, ONNX fp32 | cpu | 0.865 | 0.865 | 0.945 | 0.912 | 2.7 | 79.2 | 403.9 | 91 | 991.7x |
| student, ONNX int8 | cpu | 0.865 | 0.865 | 0.945 | 0.909 | 1.2 | 37.5 | 852.2 | 23 | 2252.5x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 10003 training texts, plus cross-entropy on 616 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.168 | 0.865 | 0.865 | 1.2 |
| 10% | 0.680 | 0.877 | 0.876 | 265.6 |
| 20% | 0.833 | 0.878 | 0.876 | 529.9 |
| 30% | 0.900 | 0.874 | 0.872 | 794.3 |
| 50% | 0.942 | 0.870 | 0.869 | 1323.1 |
| 100% | 1.000 | 0.870 | 0.869 | 2645.0 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit on cuda.
