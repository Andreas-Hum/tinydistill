| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, 8-shot | cpu | 0.314 | 0.288 | n/a | 1060.5 | 31056.6 | 1.0 | 6175 | 1.0x |
| teacher, 8-shot | cuda | 0.314 | 0.288 | n/a | 28.2 | 130.3 | 245.5 | 3087 | 37.6x |
| student, PyTorch | cpu | 0.295 ± 0.003 | 0.276 ± 0.003 | 0.870 ± 0.006 | 6.0 | 165.1 | 193.8 | 91 | 178.1x |
| student, ONNX fp32 | cpu | 0.293 | 0.273 | 0.876 | 3.7 | 163.0 | 196.3 | 91 | 286.2x |
| student, ONNX int8 | cpu | 0.276 | 0.245 | 0.878 | 1.8 | 77.2 | 414.7 | 23 | 604.7x |
| baseline: student on 8 gold labels per class, no teacher | cpu | 0.335 ± 0.069 | 0.309 ± 0.038 | 0.390 ± 0.134 | n/a | n/a | n/a | n/a | n/a |
| upper bound: student on all gold labels | cpu | 0.874 ± 0.005 | 0.832 ± 0.007 | 0.320 ± 0.003 | n/a | n/a | n/a | n/a | n/a |

Task `financial-tweets`, 2388 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows 8 gold training examples per class (24 texts). The student learns from the teacher's soft labels on all 9543 training texts, plus cross-entropy on 8 gold labels per class (24 texts, weight 1.0). The baseline is the same student trained with cross-entropy on those 24 gold labels only, without the teacher. The upper bound is the same student trained with cross-entropy on all 9543 gold training labels and no teacher; its speed equals the student's. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 8-shot on cpu.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.388 | 0.276 | 0.245 | 1.8 |
| 10% | 0.585 | 0.280 | 0.247 | 4.6 |
| 20% | 0.687 | 0.288 | 0.256 | 7.4 |
| 30% | 0.753 | 0.294 | 0.267 | 10.2 |
| 50% | 0.871 | 0.302 | 0.277 | 15.9 |
| 100% | 1.000 | 0.314 | 0.288 | 30.0 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 8-shot on cuda.
