| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.851 | 0.850 | 0.947 | n/a | 167.2 | 5275.8 | 6.1 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.826 ± 0.011 | 0.820 ± 0.012 | 0.948 ± 0.002 | 0.830 ± 0.012 | 4.6 | 83.7 | 382.4 | 91 | 36.5x |
| student, ONNX fp32 | cpu | 0.816 | 0.809 | 0.950 | 0.818 | 2.8 | 82.2 | 389.4 | 91 | 59.2x |
| student, ONNX int8 | cpu | 0.807 | 0.799 | 0.950 | 0.813 | 1.2 | 37.0 | 864.0 | 23 | 136.3x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 5000 training texts, plus cross-entropy on 616 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.087 | 0.807 | 0.799 | 1.2 |
| 10% | 0.295 | 0.825 | 0.819 | 17.9 |
| 20% | 0.361 | 0.845 | 0.842 | 34.7 |
| 30% | 0.411 | 0.853 | 0.852 | 51.4 |
| 50% | 0.517 | 0.858 | 0.857 | 84.8 |
| 100% | 1.000 | 0.851 | 0.850 | 168.5 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit on cuda.
