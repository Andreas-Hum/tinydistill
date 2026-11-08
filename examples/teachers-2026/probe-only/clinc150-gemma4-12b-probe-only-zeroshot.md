| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit, probe only | cuda | 0.891 | 0.914 | 0.935 | n/a | 186.7 | 1268.9 | 25.2 | 7504 | 1.0x |
| student, PyTorch | cpu | 0.842 ± 0.004 | 0.889 ± 0.002 | 0.900 ± 0.004 | 0.842 ± 0.005 | 4.8 | 58.9 | 543.1 | 91 | 39.2x |
| student, ONNX fp32 | cpu | 0.838 | 0.887 | 0.896 | 0.836 | 3.2 | 58.9 | 543.0 | 91 | 57.6x |
| student, ONNX int8 | cpu | 0.837 | 0.888 | 0.893 | 0.834 | 1.0 | 26.8 | 1193.9 | 23 | 190.6x |

Task `clinc150`, 5500 gold-labelled test texts. Teacher: `google/gemma-4-12B-it`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher is a linear probe on its hidden states, fit on 1208 gold training texts; fit on 966 of them, teacher accuracy on the other 242 is 94.6%. The student learns from the teacher's soft labels on all 15250 training texts, plus cross-entropy on 1208 gold-labelled texts (8 per class; weight 1.0). Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 10. The teacher scores only the 20 labels most similar to each text and always oos; the gold label was among them for 100.0% of test texts. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot, 4-bit, probe only on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.075 | 0.837 | 0.888 | 1.0 |
| 10% | 0.435 | 0.881 | 0.912 | 19.6 |
| 20% | 0.738 | 0.899 | 0.924 | 38.3 |
| 30% | 0.894 | 0.902 | 0.926 | 57.0 |
| 50% | 0.954 | 0.894 | 0.917 | 94.3 |
| 100% | 1.000 | 0.891 | 0.914 | 187.7 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot, 4-bit, probe only on cuda.
