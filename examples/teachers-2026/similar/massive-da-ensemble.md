| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher ensemble of 2, 16 similar examples, 4-bit | cuda | 0.806 | 0.779 | 0.910 | n/a | 2876.9 | n/a | 0.3 | 15034 | 1.0x |
| teacher, 16 similar examples, 4-bit: Qwen3.5-9B | cuda | 0.781 | 0.756 | 0.896 | n/a | 794.0 | 22700.2 | 1.4 | 7530 | 3.6x |
| teacher, 16 similar examples, 4-bit: gemma-4-12B-it | cuda | 0.798 | 0.771 | 0.887 | n/a | 2083.0 | n/a | 0.5 | 7504 | 1.4x |
| student, PyTorch | cpu | 0.771 ± 0.003 | 0.710 ± 0.012 | 0.897 ± 0.005 | 0.807 ± 0.003 | 3.5 | 61.6 | 519.7 | 91 | 831.0x |
| student, ONNX fp32 | cpu | 0.770 | 0.700 | 0.898 | 0.808 | 2.3 | 61.9 | 517.4 | 91 | 1264.5x |
| student, ONNX int8 | cpu | 0.764 | 0.686 | 0.893 | 0.803 | 0.8 | 29.1 | 1100.0 | 23 | 3468.3x |

Task `massive-da`, 2974 gold-labelled test texts. Teacher: `Qwen/Qwen3.5-9B,google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 476 gold training examples. The student learns from the teacher's soft labels on all 11514 training texts, plus cross-entropy on 476 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher ensemble of 2, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.128 | 0.764 | 0.686 | 0.8 |
| 10% | 0.420 | 0.790 | 0.756 | 288.1 |
| 20% | 0.569 | 0.799 | 0.775 | 576.4 |
| 30% | 0.717 | 0.805 | 0.779 | 863.7 |
| 50% | 0.887 | 0.808 | 0.780 | 1439.3 |
| 100% | 1.000 | 0.806 | 0.779 | 2877.8 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher ensemble of 2, 16 similar examples, 4-bit on cuda.
