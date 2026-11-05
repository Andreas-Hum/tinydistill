| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 8-shot, 4-bit | cuda:0 | 0.736 | 0.688 | 0.889 | n/a | n/a | n/a | n/a | n/a | n/a |
| student, PyTorch | cpu | 0.761 ± 0.001 | 0.708 ± 0.003 | 0.904 ± 0.002 | 0.802 ± 0.006 | 6.5 | 95.8 | 333.9 | 95 | n/a |
| student, ONNX fp32 | cpu | 0.760 | 0.705 | 0.904 | 0.798 | 4.2 | 90.7 | 352.7 | 96 | n/a |
| student, ONNX int8 | cpu | 0.759 | 0.706 | 0.900 | 0.798 | 1.8 | 40.2 | 797.0 | 24 | n/a |

Task `massive-da`, 2974 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. The teacher's prompt shows 8 gold training examples per class (476 texts). The student learns from the teacher's soft labels on all 11514 training texts, plus cross-entropy on 476 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. 

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.151 | 0.759 | 0.706 | n/a |
| 10% | 0.520 | 0.760 | 0.708 | n/a |
| 20% | 0.688 | 0.759 | 0.707 | n/a |
| 30% | 0.834 | 0.752 | 0.698 | n/a |
| 50% | 0.939 | 0.743 | 0.691 | n/a |
| 100% | 1.000 | 0.736 | 0.688 | n/a |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher.
