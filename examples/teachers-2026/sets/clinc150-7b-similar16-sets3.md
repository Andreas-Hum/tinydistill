| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 3 example sets, 4-bit | cuda | 0.702 | 0.761 | 0.844 | n/a | 282.9 | 6840.5 | 4.7 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.749 ± 0.005 | 0.816 ± 0.003 | 0.830 ± 0.001 | 0.734 ± 0.004 | 3.7 | 47.6 | 672.9 | 91 | 76.3x |
| student, ONNX fp32 | cpu | 0.743 | 0.813 | 0.828 | 0.730 | 2.2 | 45.4 | 705.2 | 91 | 127.7x |
| student, ONNX int8 | cpu | 0.740 | 0.809 | 0.829 | 0.725 | 0.9 | 23.0 | 1394.2 | 23 | 319.6x |

Task `clinc150`, 5500 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher scores each text 3 times, each prompt showing it another 16 of its 48 most similar of 1208 gold training examples, and averages the label probabilities. The student learns from the teacher's soft labels on all 15250 training texts, plus cross-entropy on 1208 gold-labelled texts (8 per class; weight 1.0). Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 10. The teacher scores only the 20 labels most similar to each text and always oos; the gold label was among them for 96.8% of test texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 3 example sets, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.059 | 0.740 | 0.809 | 0.9 |
| 10% | 0.293 | 0.747 | 0.812 | 29.2 |
| 20% | 0.475 | 0.749 | 0.813 | 57.5 |
| 30% | 0.591 | 0.739 | 0.803 | 85.7 |
| 50% | 0.797 | 0.712 | 0.772 | 142.3 |
| 100% | 1.000 | 0.702 | 0.761 | 283.7 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 3 example sets, 4-bit on cuda.
