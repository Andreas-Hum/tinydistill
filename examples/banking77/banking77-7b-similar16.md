| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.851 | 0.850 | 0.947 | n/a | 156.8 | 4907.8 | 6.5 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.857 ± 0.001 | 0.856 ± 0.001 | 0.947 ± 0.002 | 0.878 ± 0.003 | 3.7 | 72.7 | 440.4 | 91 | 42.6x |
| student, ONNX fp32 | cpu | 0.858 | 0.857 | 0.947 | 0.881 | 2.6 | 72.0 | 444.5 | 91 | 60.8x |
| student, ONNX int8 | cpu | 0.855 | 0.854 | 0.951 | 0.875 | 1.0 | 34.1 | 939.6 | 23 | 153.6x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 10003 training texts, plus cross-entropy on 616 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.138 | 0.855 | 0.854 | 1.0 |
| 10% | 0.622 | 0.862 | 0.861 | 16.7 |
| 20% | 0.785 | 0.862 | 0.860 | 32.4 |
| 30% | 0.875 | 0.858 | 0.857 | 48.1 |
| 50% | 0.927 | 0.855 | 0.853 | 79.4 |
| 100% | 1.000 | 0.851 | 0.850 | 157.8 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit on cuda.
