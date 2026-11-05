| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 3 example sets, 4-bit | cuda | 0.749 | 0.744 | 0.939 | n/a | 477.0 | 14887.3 | 2.1 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.809 ± 0.002 | 0.801 ± 0.002 | 0.945 ± 0.000 | 0.795 ± 0.003 | 3.8 | 72.7 | 440.0 | 91 | 124.0x |
| student, ONNX fp32 | cpu | 0.811 | 0.802 | 0.945 | 0.799 | 2.6 | 71.5 | 447.8 | 91 | 184.7x |
| student, ONNX int8 | cpu | 0.809 | 0.802 | 0.944 | 0.791 | 0.9 | 34.0 | 942.3 | 23 | 504.2x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher scores each text 3 times, each prompt showing it another 16 of its 48 most similar of 616 gold training examples, and averages the label probabilities. The student learns from the teacher's soft labels on all 10003 training texts, plus cross-entropy on 616 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 3 example sets, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.102 | 0.809 | 0.802 | 0.9 |
| 10% | 0.443 | 0.811 | 0.804 | 48.6 |
| 20% | 0.527 | 0.800 | 0.792 | 96.4 |
| 30% | 0.617 | 0.790 | 0.783 | 144.1 |
| 50% | 0.773 | 0.764 | 0.758 | 239.5 |
| 100% | 1.000 | 0.749 | 0.744 | 478.0 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 3 example sets, 4-bit on cuda.
