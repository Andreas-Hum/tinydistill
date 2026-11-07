| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher ensemble of 2, 16 similar examples, 4-bit | cuda:0 | 0.806 | 0.779 | 0.910 | n/a | n/a | n/a | n/a | 23920 | n/a |
| teacher, 16 similar examples, 4-bit: Qwen3.5-9B | cuda:0 | 0.781 | 0.756 | 0.896 | n/a | n/a | n/a | n/a | n/a | n/a |
| teacher, 16 similar examples, 4-bit: gemma-4-12B-it | cuda:0 | 0.798 | 0.771 | 0.887 | n/a | n/a | n/a | n/a | 23920 | n/a |
| student, PyTorch | cpu | 0.811 ± 0.002 | 0.783 ± 0.002 | 0.922 ± 0.002 | 0.867 ± 0.003 | 8.3 | 136.8 | 233.9 | 95 | n/a |
| student, ONNX fp32 | cpu | 0.811 | 0.781 | 0.921 | 0.863 | 5.8 | 125.0 | 256.0 | 96 | n/a |
| student, ONNX int8 | cpu | 0.810 | 0.787 | 0.921 | 0.857 | 1.9 | 46.2 | 692.8 | 24 | n/a |

Task `massive-da`, 2974 gold-labelled test texts. Teacher: `Qwen/Qwen3.5-9B,google/gemma-4-12B-it`. Student: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. The teacher's prompt shows each text the 16 most similar of 476 gold training examples. The student learns from the teacher's soft labels on all 11514 training texts, plus cross-entropy on 476 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. 

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.129 | 0.810 | 0.787 | n/a |
| 10% | 0.546 | 0.814 | 0.789 | n/a |
| 20% | 0.734 | 0.817 | 0.794 | n/a |
| 30% | 0.864 | 0.816 | 0.794 | n/a |
| 50% | 0.951 | 0.807 | 0.781 | n/a |
| 100% | 1.000 | 0.806 | 0.779 | n/a |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher.
