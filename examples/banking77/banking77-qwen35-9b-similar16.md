| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.861 | 0.860 | 0.952 | n/a | 907.3 | 26155.5 | 1.2 | 7530 | 1.0x |
| student, PyTorch | cpu | 0.867 ± 0.004 | 0.867 ± 0.004 | 0.954 ± 0.002 | 0.905 ± 0.001 | 6.1 | 117.4 | 272.6 | 91 | 147.6x |
| student, ONNX fp32 | cpu | 0.864 | 0.865 | 0.951 | 0.905 | 4.4 | 123.5 | 259.2 | 91 | 204.5x |
| student, ONNX int8 | cpu | 0.859 | 0.859 | 0.951 | 0.901 | 1.4 | 52.9 | 604.5 | 23 | 636.1x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen3.5-9B`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 10003 training texts, plus cross-entropy on 616 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.127 | 0.859 | 0.859 | 1.4 |
| 10% | 0.600 | 0.872 | 0.872 | 92.2 |
| 20% | 0.774 | 0.874 | 0.873 | 182.9 |
| 30% | 0.856 | 0.870 | 0.869 | 273.6 |
| 50% | 0.923 | 0.865 | 0.865 | 455.1 |
| 100% | 1.000 | 0.861 | 0.860 | 908.7 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit on cuda.
