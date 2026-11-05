| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | In-scope accuracy | Out-of-scope recall | Out-of-scope precision | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.769 | 0.834 | 0.824 | 0.912 | 0.126 | 0.969 | n/a | 97.4 | 2267.9 | 14.1 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.767 ± 0.002 | 0.832 ± 0.001 | 0.830 ± 0.002 | 0.916 ± 0.001 | 0.098 ± 0.009 | 0.989 ± 0.011 | 0.791 ± 0.001 | 6.3 | 95.7 | 334.4 | 91 | 15.5x |
| student, ONNX fp32 | cpu | 0.769 | 0.833 | 0.830 | 0.916 | 0.106 | 1.000 | 0.790 | 3.9 | 104.7 | 305.6 | 91 | 25.0x |
| student, ONNX int8 | cpu | 0.767 | 0.832 | 0.829 | 0.915 | 0.105 | 0.981 | 0.791 | 1.6 | 42.5 | 753.8 | 23 | 61.6x |

Task `clinc150-outside`, 5500 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 1208 gold training examples. The student learns from the teacher's soft labels on all 15250 training texts, plus cross-entropy on 1208 gold-labelled texts (8 per class; weight 1.0). Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 10. The teacher scores only the 20 labels most similar to each text and always oos; the gold label was among them for 96.8% of test texts. In-scope accuracy is the accuracy on the test texts whose gold label is not the `outside` label (texts that fit none of the others); out-of-scope recall and precision are those of that label. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.067 | 0.767 | 0.832 | 1.6 |
| 10% | 0.376 | 0.773 | 0.836 | 11.3 |
| 20% | 0.658 | 0.777 | 0.841 | 21.1 |
| 30% | 0.876 | 0.776 | 0.841 | 30.8 |
| 50% | 0.958 | 0.770 | 0.835 | 50.3 |
| 100% | 1.000 | 0.769 | 0.834 | 99.0 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit on cuda.
