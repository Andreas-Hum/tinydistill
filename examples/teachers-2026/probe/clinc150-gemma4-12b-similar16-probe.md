| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe | cuda | 0.941 | 0.950 | 0.971 | n/a | 1299.5 | 38430.5 | 0.8 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.860 ± 0.005 | 0.898 ± 0.004 | 0.915 ± 0.002 | 0.870 ± 0.004 | 3.4 | 47.4 | 674.7 | 91 | 384.2x |
| student, ONNX fp32 | cpu | 0.861 | 0.898 | 0.916 | 0.870 | 2.0 | 45.5 | 703.1 | 91 | 639.3x |
| student, ONNX int8 | cpu | 0.859 | 0.896 | 0.917 | 0.868 | 0.8 | 22.4 | 1429.8 | 23 | 1540.1x |

Task `clinc150`, 5500 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's probabilities are averaged with those of a linear probe on its hidden states, fit on 1208 gold training texts; fit on 966 of them, teacher accuracy on the other 242 is 94.6% in context, 98.3% for the probe and 98.8% combined. The teacher's prompt shows each text the 16 most similar of 1208 gold training examples. The student learns from the teacher's soft labels on all 15250 training texts, plus cross-entropy on 1208 gold-labelled texts (8 per class; weight 1.0). Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 10. The teacher scores only the 20 labels most similar to each text and always oos; the gold label was among them for 100.0% of test texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.074 | 0.859 | 0.896 | 0.8 |
| 10% | 0.505 | 0.911 | 0.929 | 130.8 |
| 20% | 0.806 | 0.933 | 0.946 | 260.7 |
| 30% | 0.922 | 0.938 | 0.949 | 390.7 |
| 50% | 0.968 | 0.942 | 0.951 | 650.6 |
| 100% | 1.000 | 0.941 | 0.950 | 1300.4 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe on cuda.
