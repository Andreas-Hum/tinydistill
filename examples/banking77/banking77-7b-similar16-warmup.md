| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.851 | 0.850 | 0.947 | n/a | n/a | n/a | n/a | n/a | n/a |
| student, PyTorch | cpu | 0.855 ± 0.002 | 0.854 ± 0.002 | 0.947 ± 0.001 | 0.875 ± 0.002 | 5.5 | 146.8 | 218.0 | 91 | n/a |
| student, ONNX fp32 | cpu | 0.855 | 0.854 | 0.945 | 0.873 | 3.4 | 116.0 | 275.9 | 91 | n/a |
| student, ONNX int8 | cpu | 0.850 | 0.849 | 0.946 | 0.872 | 1.6 | 55.8 | 573.7 | 23 | n/a |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 10003 training texts, plus cross-entropy on 616 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. 

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.163 | 0.850 | 0.849 | n/a |
| 10% | 0.612 | 0.864 | 0.862 | n/a |
| 20% | 0.794 | 0.861 | 0.860 | n/a |
| 30% | 0.873 | 0.859 | 0.857 | n/a |
| 50% | 0.929 | 0.854 | 0.852 | n/a |
| 100% | 1.000 | 0.851 | 0.850 | n/a |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher.
