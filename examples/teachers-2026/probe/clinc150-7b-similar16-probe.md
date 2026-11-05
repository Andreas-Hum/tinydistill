| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe | cuda | 0.842 | 0.887 | 0.957 | n/a | 92.0 | 2267.3 | 14.1 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.817 ± 0.007 | 0.873 ± 0.004 | 0.880 ± 0.003 | 0.832 ± 0.007 | 3.6 | 47.0 | 680.5 | 91 | 25.3x |
| student, ONNX fp32 | cpu | 0.810 | 0.869 | 0.877 | 0.824 | 2.4 | 45.4 | 704.6 | 91 | 38.2x |
| student, ONNX int8 | cpu | 0.813 | 0.872 | 0.878 | 0.824 | 0.9 | 22.1 | 1445.0 | 23 | 108.0x |

Task `clinc150`, 5500 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's probabilities are averaged with those of a linear probe on its hidden states, fit on 1208 gold training texts; fit on 966 of them, teacher accuracy on the other 242 is 94.6% in context, 96.7% for the probe and 95.5% combined. The teacher's prompt shows each text the 16 most similar of 1208 gold training examples. The student learns from the teacher's soft labels on all 15250 training texts, plus cross-entropy on 1208 gold-labelled texts (8 per class; weight 1.0). Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 9, 10. The teacher scores only the 20 labels most similar to each text and always oos; the gold label was among them for 100.0% of test texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.077 | 0.813 | 0.872 | 0.9 |
| 10% | 0.406 | 0.836 | 0.884 | 10.0 |
| 20% | 0.674 | 0.845 | 0.892 | 19.2 |
| 30% | 0.876 | 0.844 | 0.890 | 28.4 |
| 50% | 0.961 | 0.842 | 0.887 | 46.8 |
| 100% | 1.000 | 0.842 | 0.887 | 92.8 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe on cuda.
