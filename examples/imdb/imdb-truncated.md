| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot | cuda | 0.934 | 0.934 | n/a | 27.2 | 284.1 | 112.6 | 3087 | 1.0x |
| student, PyTorch | cpu | 0.844 ± 0.007 | 0.843 ± 0.007 | 0.851 ± 0.005 | 13.3 | 342.9 | 93.3 | 91 | 2.0x |
| student, ONNX fp32 | cpu | 0.842 | 0.842 | 0.853 | 12.6 | 405.6 | 78.9 | 91 | 2.2x |
| student, ONNX int8 | cpu | 0.836 | 0.836 | 0.848 | 6.4 | 217.4 | 147.2 | 23 | 4.3x |

Task `imdb`, 2000 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 5000 unlabelled training texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.502 | 0.836 | 0.836 | 6.4 |
| 10% | 0.803 | 0.875 | 0.875 | 9.1 |
| 20% | 0.936 | 0.896 | 0.896 | 11.8 |
| 30% | 0.982 | 0.914 | 0.914 | 14.6 |
| 50% | 0.996 | 0.930 | 0.930 | 20.0 |
| 100% | 1.000 | 0.934 | 0.934 | 33.6 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot on cuda.
