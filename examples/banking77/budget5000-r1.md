| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.851 | 0.850 | 0.947 | n/a | 163.8 | 5294.8 | 6.0 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.830 ± 0.012 | 0.823 ± 0.013 | 0.945 ± 0.006 | 0.842 ± 0.007 | 4.4 | 81.7 | 391.6 | 91 | 37.1x |
| student, ONNX fp32 | cpu | 0.830 | 0.823 | 0.950 | 0.838 | 3.1 | 79.7 | 401.4 | 91 | 52.2x |
| student, ONNX int8 | cpu | 0.821 | 0.813 | 0.947 | 0.834 | 1.2 | 38.9 | 821.6 | 23 | 136.4x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 5000 training texts, plus cross-entropy on 616 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.099 | 0.821 | 0.813 | 1.2 |
| 10% | 0.319 | 0.842 | 0.837 | 17.6 |
| 20% | 0.411 | 0.855 | 0.852 | 34.0 |
| 30% | 0.482 | 0.862 | 0.861 | 50.3 |
| 50% | 0.580 | 0.859 | 0.858 | 83.1 |
| 100% | 1.000 | 0.851 | 0.850 | 165.0 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit on cuda.
