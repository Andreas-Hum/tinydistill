| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher ensemble of 3, zero-shot, 4-bit | cuda | 0.842 | 0.801 | n/a | 241.6 | 1926.9 | 16.6 | 11677 | 1.0x |
| teacher, zero-shot, 4-bit: Qwen2.5-7B-Instruct | cuda | 0.832 | 0.788 | n/a | 109.4 | 774.0 | 41.3 | 5443 | 2.2x |
| teacher, zero-shot, 4-bit: Mistral-7B-Instruct-v0.3 | cuda | 0.832 | 0.781 | n/a | 75.1 | 599.6 | 53.4 | 4027 | 3.2x |
| teacher, zero-shot, 4-bit: Phi-3.5-mini-instruct | cuda | 0.800 | 0.765 | n/a | 57.2 | 553.3 | 57.8 | 2206 | 4.2x |
| student, PyTorch | cpu | 0.828 ± 0.006 | 0.778 ± 0.005 | 0.884 ± 0.007 | 6.5 | 207.4 | 154.3 | 91 | 37.5x |
| student, ONNX fp32 | cpu | 0.826 | 0.776 | 0.891 | 3.3 | 196.3 | 163.0 | 91 | 73.2x |
| student, ONNX int8 | cpu | 0.829 | 0.780 | 0.879 | 1.6 | 99.4 | 322.1 | 23 | 148.9x |
| upper bound: student on all gold labels | cpu | 0.874 ± 0.005 | 0.832 ± 0.007 | 0.815 ± 0.002 | n/a | n/a | n/a | n/a | n/a |

Task `financial-tweets`, 2388 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3,microsoft/Phi-3.5-mini-instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 9543 unlabelled training texts. The upper bound is the same student trained with cross-entropy on all 9543 gold training labels and no teacher; its speed equals the student's. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher ensemble of 3, zero-shot, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.377 | 0.829 | 0.780 | 1.6 |
| 10% | 0.646 | 0.832 | 0.785 | 25.8 |
| 20% | 0.772 | 0.834 | 0.788 | 50.0 |
| 30% | 0.876 | 0.839 | 0.796 | 74.1 |
| 50% | 0.977 | 0.843 | 0.802 | 122.4 |
| 100% | 1.000 | 0.842 | 0.801 | 243.3 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher ensemble of 3, zero-shot, 4-bit on cuda.
