| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit | cuda | 0.555 | 0.608 | 0.709 | n/a | 72.8 | 1248.8 | 25.6 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.556 ± 0.001 | 0.600 ± 0.002 | 0.714 ± 0.004 | 0.726 ± 0.002 | 4.5 | 58.8 | 544.1 | 91 | 16.2x |
| student, ONNX fp32 | cpu | 0.556 | 0.601 | 0.709 | 0.726 | 2.9 | 49.7 | 643.8 | 91 | 24.8x |
| student, ONNX int8 | cpu | 0.555 | 0.601 | 0.712 | 0.723 | 1.7 | 28.2 | 1136.3 | 23 | 43.8x |

Task `clinc150`, 5500 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 15250 unlabelled training texts. Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 10. The teacher scores only the 20 labels most similar to each text; the gold label was among them for 80.3% of test texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.071 | 0.555 | 0.601 | 1.7 |
| 10% | 0.359 | 0.560 | 0.607 | 8.9 |
| 20% | 0.528 | 0.562 | 0.608 | 16.2 |
| 30% | 0.670 | 0.560 | 0.606 | 23.5 |
| 50% | 0.890 | 0.558 | 0.608 | 38.1 |
| 100% | 1.000 | 0.555 | 0.608 | 74.5 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot, 4-bit on cuda.
