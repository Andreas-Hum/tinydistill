| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot | cuda | 0.875 | 0.873 | n/a | 26.0 | 282.1 | 113.4 | 3087 | 1.0x |
| student, PyTorch | cpu | 0.860 ± 0.006 | 0.858 ± 0.006 | 0.926 ± 0.001 | 36.4 | 1347.2 | 23.8 | 91 | 0.7x |
| student, ONNX fp32 | cpu | 0.867 | 0.865 | 0.927 | 35.7 | 1524.4 | 21.0 | 91 | 0.7x |
| student, ONNX int8 | cpu | 0.858 | 0.855 | 0.924 | 18.0 | 866.0 | 37.0 | 23 | 1.4x |

Task `imdb`, 2000 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 21523 unlabelled training texts. Texts longer than 128 tokens are read in windows of that many tokens, 64 apart: the training texts are windows, and a test text's prediction is the mean of its windows' probabilities. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.500 | 0.858 | 0.855 | 18.0 |
| 10% | 0.617 | 0.870 | 0.868 | 20.6 |
| 20% | 0.728 | 0.876 | 0.874 | 23.2 |
| 30% | 0.835 | 0.878 | 0.877 | 25.8 |
| 50% | 0.977 | 0.875 | 0.874 | 31.0 |
| 100% | 1.000 | 0.875 | 0.873 | 44.0 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot on cuda.
