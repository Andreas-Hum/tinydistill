| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.798 | 0.771 | 0.887 | n/a | 2335.7 | n/a | 0.4 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.760 ± 0.003 | 0.706 ± 0.009 | 0.883 ± 0.006 | 0.796 ± 0.005 | 3.6 | 62.1 | 514.9 | 91 | 648.7x |
| student, ONNX fp32 | cpu | 0.757 | 0.700 | 0.889 | 0.792 | 2.4 | 60.2 | 531.8 | 91 | 993.1x |
| student, ONNX int8 | cpu | 0.753 | 0.685 | 0.884 | 0.794 | 0.9 | 28.8 | 1112.7 | 23 | 2712.6x |

Task `massive-da`, 2974 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 476 gold training examples. The student learns from the teacher's soft labels on all 11514 training texts, plus cross-entropy on 476 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.134 | 0.753 | 0.685 | 0.9 |
| 10% | 0.441 | 0.783 | 0.752 | 234.1 |
| 20% | 0.606 | 0.796 | 0.768 | 468.2 |
| 30% | 0.763 | 0.801 | 0.775 | 701.4 |
| 50% | 0.912 | 0.801 | 0.775 | 1168.7 |
| 100% | 1.000 | 0.798 | 0.771 | 2336.6 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit on cuda.
