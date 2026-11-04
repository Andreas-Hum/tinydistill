| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 8-shot, 4-bit | cuda | 0.736 | 0.688 | 0.889 | n/a | 143.8 | 4171.0 | 7.7 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.724 ± 0.006 | 0.667 ± 0.004 | 0.866 ± 0.004 | 0.752 ± 0.006 | 4.1 | 66.0 | 484.9 | 91 | 35.0x |
| student, ONNX fp32 | cpu | 0.720 | 0.667 | 0.870 | 0.744 | 2.2 | 62.2 | 514.9 | 91 | 64.9x |
| student, ONNX int8 | cpu | 0.713 | 0.663 | 0.871 | 0.747 | 0.9 | 29.5 | 1083.8 | 23 | 165.5x |
| baseline: student on 8 gold labels per class, no teacher | cpu | 0.350 ± 0.028 | 0.318 ± 0.007 | 0.586 ± 0.029 | 0.340 ± 0.024 | n/a | n/a | n/a | n/a | n/a |
| upper bound: student on all gold labels | cpu | 0.820 ± 0.006 | 0.736 ± 0.010 | 0.914 ± 0.004 | 0.703 ± 0.003 | n/a | n/a | n/a | n/a | n/a |

Task `massive-da`, 2974 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows 8 gold training examples per class (476 texts). The student learns from the teacher's soft labels on all 11514 training texts, plus cross-entropy on 476 gold-labelled texts (8 per class; weight 1.0). The baseline is the same student trained with cross-entropy on those 476 gold labels only, without the teacher. The upper bound is the same student trained with cross-entropy on all 11514 gold training labels and no teacher; its speed equals the student's. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 8-shot, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.115 | 0.713 | 0.663 | 0.9 |
| 10% | 0.369 | 0.733 | 0.674 | 15.2 |
| 20% | 0.501 | 0.746 | 0.694 | 29.6 |
| 30% | 0.630 | 0.748 | 0.695 | 44.0 |
| 50% | 0.831 | 0.741 | 0.690 | 72.8 |
| 100% | 1.000 | 0.736 | 0.688 | 144.7 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 8-shot, 4-bit on cuda.
