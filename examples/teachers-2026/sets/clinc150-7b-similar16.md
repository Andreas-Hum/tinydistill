| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.775 | 0.837 | 0.832 | n/a | 92.6 | 2256.8 | 14.2 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.767 ± 0.006 | 0.834 ± 0.003 | 0.826 ± 0.003 | 0.796 ± 0.004 | 3.8 | 49.6 | 645.5 | 91 | 24.7x |
| student, ONNX fp32 | cpu | 0.763 | 0.833 | 0.826 | 0.793 | 2.3 | 45.4 | 704.6 | 91 | 40.6x |
| student, ONNX int8 | cpu | 0.760 | 0.830 | 0.823 | 0.795 | 0.8 | 22.2 | 1439.6 | 23 | 111.5x |

Task `clinc150`, 5500 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 1208 gold training examples. The student learns from the teacher's soft labels on all 15250 training texts, plus cross-entropy on 1208 gold-labelled texts (8 per class; weight 1.0). Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 10. The teacher scores only the 20 labels most similar to each text and always oos; the gold label was among them for 96.8% of test texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.072 | 0.760 | 0.830 | 0.8 |
| 10% | 0.364 | 0.773 | 0.838 | 10.1 |
| 20% | 0.652 | 0.779 | 0.843 | 19.3 |
| 30% | 0.885 | 0.777 | 0.841 | 28.6 |
| 50% | 0.962 | 0.775 | 0.838 | 47.1 |
| 100% | 1.000 | 0.775 | 0.837 | 93.4 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit on cuda.
