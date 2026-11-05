| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 2 prompts | cuda | 0.770 | 0.774 | n/a | 56.9 | 272.4 | 117.5 | 3087 | 1.0x |
| student, PyTorch | cpu | 0.796 ± 0.005 | 0.798 ± 0.005 | 0.852 ± 0.001 | 11.0 | 563.7 | 56.8 | 91 | 5.2x |
| student, ONNX fp32 | cpu | 0.799 | 0.801 | 0.853 | 11.1 | 569.4 | 56.2 | 91 | 5.1x |
| student, ONNX int8 | cpu | 0.797 | 0.800 | 0.851 | 3.8 | 308.8 | 103.6 | 23 | 15.1x |

Task `ag-news`, 7600 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 10000 unlabelled training texts. Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 7, 8. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, 2 prompts on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.277 | 0.797 | 0.800 | 3.8 |
| 10% | 0.480 | 0.800 | 0.803 | 9.5 |
| 20% | 0.546 | 0.795 | 0.797 | 15.2 |
| 30% | 0.602 | 0.787 | 0.790 | 20.8 |
| 50% | 0.718 | 0.777 | 0.780 | 32.2 |
| 100% | 1.000 | 0.770 | 0.774 | 60.7 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot, 2 prompts on cuda.
