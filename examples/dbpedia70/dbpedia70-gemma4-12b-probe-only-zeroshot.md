| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit, probe only | cuda | 0.813 | 0.804 | 0.926 | n/a | 171.5 | 3434.2 | 9.3 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.828 ± 0.001 | 0.823 ± 0.002 | 0.939 ± 0.001 | 0.907 ± 0.001 | 15.1 | 398.7 | 80.3 | 91 | 11.3x |
| student, ONNX fp32 | cpu | 0.829 | 0.825 | 0.939 | 0.908 | 16.3 | 733.1 | 43.7 | 91 | 10.5x |
| student, ONNX int8 | cpu | 0.829 | 0.825 | 0.939 | 0.907 | 10.7 | 655.8 | 48.8 | 23 | 16.0x |
| baseline: student on 8 gold labels per class, no teacher | cpu | 0.593 ± 0.006 | 0.611 ± 0.005 | 0.787 ± 0.024 | 0.625 ± 0.009 | n/a | n/a | n/a | n/a | n/a |

Task `dbpedia70`, 60794 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 560 gold training texts; fit on 448 of them, teacher accuracy on the other 112 is 91.1%. The student learns from the teacher's soft labels on all 30000 training texts, plus cross-entropy on 560 gold-labelled texts (8 per class; weight 1.0). The baseline is the same student trained with cross-entropy on those 560 gold labels only, without the teacher. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.124 | 0.829 | 0.825 | 10.7 |
| 10% | 0.522 | 0.827 | 0.822 | 27.9 |
| 20% | 0.738 | 0.822 | 0.816 | 45.0 |
| 30% | 0.906 | 0.818 | 0.811 | 62.2 |
| 50% | 0.984 | 0.814 | 0.807 | 96.5 |
| 100% | 1.000 | 0.813 | 0.804 | 182.2 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot, 4-bit, probe only on cuda.
