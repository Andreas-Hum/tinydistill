| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit, probe only | cuda | 0.792 | 0.765 | 0.903 | n/a | 170.1 | 3627.4 | 8.8 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.804 ± 0.002 | 0.777 ± 0.004 | 0.917 ± 0.003 | 0.826 ± 0.003 | 7.0 | 92.6 | 345.4 | 95 | 24.2x |
| student, ONNX fp32 | cpu | 0.806 | 0.781 | 0.914 | 0.822 | 4.6 | 89.1 | 359.2 | 96 | 37.3x |
| student, ONNX int8 | cpu | 0.802 | 0.777 | 0.915 | 0.820 | 1.6 | 40.6 | 788.9 | 24 | 104.2x |

Task `massive-da`, 2974 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. The teacher is a linear probe on its hidden states, fit on 467 gold training texts; fit on 374 of them, teacher accuracy on the other 93 is 76.3%. The teacher's prompt shows each text the 16 most similar of 476 gold training examples. The student learns from the teacher's soft labels on all 11514 training texts, plus cross-entropy on 476 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.159 | 0.802 | 0.777 | 1.6 |
| 10% | 0.513 | 0.809 | 0.783 | 18.6 |
| 20% | 0.695 | 0.810 | 0.782 | 35.7 |
| 30% | 0.830 | 0.807 | 0.779 | 52.6 |
| 50% | 0.934 | 0.799 | 0.774 | 86.7 |
| 100% | 1.000 | 0.792 | 0.765 | 171.7 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit, probe only on cuda.
