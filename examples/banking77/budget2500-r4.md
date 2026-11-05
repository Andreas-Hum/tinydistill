| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.851 | 0.850 | 0.947 | n/a | 168.7 | 5174.6 | 6.2 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.725 ± 0.017 | 0.701 ± 0.020 | 0.908 ± 0.002 | 0.732 ± 0.014 | 5.3 | 98.6 | 324.7 | 91 | 32.1x |
| student, ONNX fp32 | cpu | 0.715 | 0.689 | 0.910 | 0.724 | 3.4 | 77.0 | 415.5 | 91 | 50.1x |
| student, ONNX int8 | cpu | 0.709 | 0.684 | 0.906 | 0.718 | 1.6 | 42.2 | 758.5 | 23 | 106.4x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 2500 training texts, plus cross-entropy on 616 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.064 | 0.709 | 0.684 | 1.6 |
| 10% | 0.135 | 0.742 | 0.723 | 18.5 |
| 20% | 0.161 | 0.769 | 0.758 | 35.3 |
| 30% | 0.186 | 0.787 | 0.777 | 52.2 |
| 50% | 0.237 | 0.809 | 0.801 | 85.9 |
| 100% | 1.000 | 0.851 | 0.850 | 170.3 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit on cuda.
