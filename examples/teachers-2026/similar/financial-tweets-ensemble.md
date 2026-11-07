| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher ensemble of 2, 16 similar examples, 4-bit | cuda | 0.859 | 0.833 | n/a | 527.2 | 15460.3 | 2.1 | 15034 | 1.0x |
| teacher, 16 similar examples, 4-bit: Qwen3.5-9B | cuda | 0.821 | 0.770 | n/a | 174.8 | 5160.6 | 6.2 | 7530 | 3.0x |
| teacher, 16 similar examples, 4-bit: gemma-4-12B-it | cuda | 0.846 | 0.823 | n/a | 352.4 | 10299.7 | 3.1 | 7504 | 1.5x |
| student, PyTorch | cpu | 0.827 ± 0.008 | 0.788 ± 0.008 | 0.853 ± 0.001 | 4.2 | 140.1 | 228.4 | 91 | 124.7x |
| student, ONNX fp32 | cpu | 0.823 | 0.782 | 0.854 | 2.8 | 144.8 | 221.0 | 91 | 188.7x |
| student, ONNX int8 | cpu | 0.829 | 0.787 | 0.847 | 1.3 | 67.6 | 473.1 | 23 | 417.4x |

Task `financial-tweets`, 2388 gold-labelled test texts. Teacher: `Qwen/Qwen3.5-9B,google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 24 gold training examples. The student learns from the teacher's soft labels on all 9543 training texts, plus cross-entropy on 24 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher ensemble of 2, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.367 | 0.829 | 0.787 | 1.3 |
| 10% | 0.583 | 0.848 | 0.811 | 54.0 |
| 20% | 0.715 | 0.853 | 0.820 | 106.8 |
| 30% | 0.819 | 0.854 | 0.823 | 159.3 |
| 50% | 0.934 | 0.860 | 0.831 | 264.9 |
| 100% | 1.000 | 0.859 | 0.833 | 528.5 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher ensemble of 2, 16 similar examples, 4-bit on cuda.
