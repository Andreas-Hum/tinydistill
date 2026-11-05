| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.851 | 0.850 | 0.947 | n/a | n/a | n/a | n/a | n/a | n/a |
| student, PyTorch | cpu | 0.739 ± 0.010 | 0.709 ± 0.012 | 0.926 ± 0.004 | 0.756 ± 0.010 | 6.3 | 158.4 | 202.0 | 91 | n/a |
| student, ONNX fp32 | cpu | 0.733 | 0.703 | 0.922 | 0.752 | 5.1 | 155.9 | 205.3 | 91 | n/a |
| student, ONNX int8 | cpu | 0.732 | 0.701 | 0.917 | 0.750 | 1.8 | 55.6 | 575.1 | 23 | n/a |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 2500 training texts, plus cross-entropy on 616 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. 

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.073 | 0.732 | 0.701 | n/a |
| 10% | 0.170 | 0.752 | 0.723 | n/a |
| 20% | 0.210 | 0.776 | 0.755 | n/a |
| 30% | 0.243 | 0.799 | 0.784 | n/a |
| 50% | 0.311 | 0.825 | 0.819 | n/a |
| 100% | 1.000 | 0.851 | 0.850 | n/a |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher.
