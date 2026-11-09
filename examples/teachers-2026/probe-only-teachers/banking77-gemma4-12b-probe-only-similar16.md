| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe only | cuda | 0.873 | 0.873 | 0.941 | n/a | 149.9 | 3725.7 | 8.6 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.870 ± 0.003 | 0.870 ± 0.003 | 0.949 ± 0.002 | 0.906 ± 0.004 | 3.9 | 78.0 | 410.4 | 91 | 38.4x |
| student, ONNX fp32 | cpu | 0.867 | 0.867 | 0.947 | 0.904 | 2.9 | 75.5 | 423.8 | 91 | 51.2x |
| student, ONNX int8 | cpu | 0.869 | 0.869 | 0.948 | 0.906 | 1.1 | 35.5 | 901.0 | 23 | 139.5x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 616 gold training texts; fit on 493 of them, teacher accuracy on the other 123 is 82.1%. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 10003 training texts, plus cross-entropy on 616 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.189 | 0.869 | 0.869 | 1.1 |
| 10% | 0.633 | 0.878 | 0.877 | 16.1 |
| 20% | 0.802 | 0.878 | 0.878 | 31.0 |
| 30% | 0.872 | 0.877 | 0.876 | 46.0 |
| 50% | 0.929 | 0.874 | 0.873 | 76.0 |
| 100% | 1.000 | 0.873 | 0.873 | 150.9 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe only on cuda.
