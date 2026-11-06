| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher ensemble of 2, 16 similar examples, 4-bit | cuda | 0.879 | 0.878 | 0.958 | n/a | 3589.6 | n/a | 0.3 | 15034 | 1.0x |
| teacher, 16 similar examples, 4-bit: Qwen3.5-9B | cuda | 0.861 | 0.860 | 0.952 | n/a | 807.7 | 26754.6 | 1.2 | 7530 | 4.4x |
| teacher, 16 similar examples, 4-bit: gemma-4-12B-it | cuda | 0.870 | 0.869 | 0.939 | n/a | 2781.8 | n/a | 0.4 | 7504 | 1.3x |
| student, PyTorch | cpu | 0.875 ± 0.003 | 0.875 ± 0.003 | 0.957 ± 0.002 | 0.924 ± 0.001 | 5.2 | 88.4 | 361.8 | 91 | 689.1x |
| student, ONNX fp32 | cpu | 0.872 | 0.872 | 0.956 | 0.923 | 3.1 | 82.0 | 390.2 | 91 | 1169.8x |
| student, ONNX int8 | cpu | 0.870 | 0.869 | 0.956 | 0.918 | 1.4 | 40.6 | 787.9 | 23 | 2637.4x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen3.5-9B,google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 10003 training texts, plus cross-entropy on 616 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher ensemble of 2, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.153 | 0.870 | 0.869 | 1.4 |
| 10% | 0.641 | 0.883 | 0.883 | 360.3 |
| 20% | 0.804 | 0.881 | 0.880 | 719.3 |
| 30% | 0.876 | 0.881 | 0.880 | 1078.2 |
| 50% | 0.933 | 0.879 | 0.878 | 1796.1 |
| 100% | 1.000 | 0.879 | 0.878 | 3590.9 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher ensemble of 2, 16 similar examples, 4-bit on cuda.
