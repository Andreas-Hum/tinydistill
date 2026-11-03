| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, 8-shot, 4-bit | cuda | 0.783 | 0.779 | n/a | 264.0 | 7434.4 | 4.3 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.729 ± 0.011 | 0.697 ± 0.015 | 0.800 ± 0.003 | 4.4 | 89.3 | 358.5 | 91 | 60.3x |
| student, ONNX fp32 | cpu | 0.727 | 0.696 | 0.801 | 3.4 | 90.9 | 352.2 | 91 | 77.6x |
| student, ONNX int8 | cpu | 0.726 | 0.694 | 0.797 | 1.4 | 40.2 | 795.7 | 23 | 186.3x |
| baseline: student on 8 gold labels per class, no teacher | cpu | 0.490 ± 0.014 | 0.449 ± 0.018 | 0.473 ± 0.015 | n/a | n/a | n/a | n/a | n/a |
| upper bound: student on all gold labels | cpu | 0.914 ± 0.005 | 0.914 ± 0.005 | 0.775 ± 0.001 | n/a | n/a | n/a | n/a | n/a |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows 8 gold training examples per class (616 texts). The student learns from the teacher's soft labels on all 10003 training texts, plus cross-entropy on 8 gold labels per class (616 texts, weight 1.0). The baseline is the same student trained with cross-entropy on those 616 gold labels only, without the teacher. The upper bound is the same student trained with cross-entropy on all 10003 gold training labels and no teacher; its speed equals the student's. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 8-shot, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.080 | 0.726 | 0.694 | 1.4 |
| 10% | 0.231 | 0.744 | 0.718 | 27.8 |
| 20% | 0.302 | 0.750 | 0.730 | 54.2 |
| 30% | 0.355 | 0.764 | 0.748 | 80.6 |
| 50% | 0.440 | 0.778 | 0.769 | 133.4 |
| 100% | 1.000 | 0.783 | 0.779 | 265.4 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 8-shot, 4-bit on cuda.
