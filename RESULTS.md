# Results

Every table on this page is written by the example scripts in [`examples/`](https://github.com/Andreas-Hum/tinydistill/tree/main/examples) from their saved
results files; the text only quotes them. The [README](https://github.com/Andreas-Hum/tinydistill#results) has a summary, and the
[live demo](https://andreas-hum.github.io/tinydistill/demo/) runs the Banking77 student in your browser.

Teachers are `Qwen/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-7B-Instruct` (4 bit), zero-shot or with 8 gold
examples per class in the prompt; the student is `sentence-transformers/all-MiniLM-L6-v2` unless a table says
otherwise. Quality is measured on gold-labelled test texts that no model trains on, and students are trained
with 3 seeds (mean ± standard deviation). Students train for 5 epochs (the default before 0.3.1) unless a
table says otherwise; see [Training the student](https://andreas-hum.github.io/tinydistill/guide/training/).

## Compared with other methods

[How it compares](https://andreas-hum.github.io/tinydistill/how-it-compares/) puts the best tinydistill student
of each task (Banking77, CLINC150, Danish MASSIVE, financial tweets) next to what people use instead, with the
same 8 gold labels per class and the same test set: the student's encoder fine-tuned on the gold labels alone,
SetFit, GLiClass zero-shot and the LLM itself, with accuracy, CPU latency, size and the cost to build. Published
10-shot results on Banking77 are listed separately. The tables are written by
[`examples/compare/`](https://github.com/Andreas-Hum/tinydistill/tree/main/examples/compare).

On Banking77 the best student reaches 0.875 against 0.796 for SetFit with the same encoder, which trains in 7
minutes on a CPU and needs no LLM; on CLINC150, 0.860 against 0.712. With a zero-shot teacher and no gold labels
the CLINC150 student (0.573) is below both SetFit and the encoder fine-tuned on the gold labels alone (0.580).

## Banking77: the hard case

Intent detection with 77 fine-grained classes ([Banking77](https://huggingface.co/datasets/PolyAI/banking77),
CC BY 4.0), reproduced by [`examples/banking77/run.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/banking77/run.sh).
One row per teacher; the student columns are the deployed int8 model:

<!-- results: banking77/teachers.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77-1.5b | zero-shot `Qwen/Qwen2.5-1.5B-Instruct` | 0.404 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.386 ± 0.004 | 0.313 ± 0.005 | 0.766 ± 0.004 | ONNX int8 | 1.2 | 37.3 | 23 |
| banking77-1.5b-8shot | 8-shot `Qwen/Qwen2.5-1.5B-Instruct` | 0.513 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.485 ± 0.009 | 0.406 ± 0.007 | 0.706 ± 0.002 | ONNX int8 | 1.1 | 36.4 | 23 |
| banking77-7b | zero-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.629 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.618 ± 0.003 | 0.579 ± 0.005 | 0.817 ± 0.007 | ONNX int8 | 1.4 | 39.2 | 23 |
| banking77-7b-8shot | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.729 ± 0.011 | 0.697 ± 0.015 | 0.800 ± 0.003 | ONNX int8 | 1.4 | 40.2 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

The student follows its teacher, so the teacher sets the ceiling: a bigger model and 8 examples per class
each help a lot. The full table for the best teacher, with the student trained on the same 616 gold labels
alone (baseline) and on all 10,003 gold labels (upper bound), and the cascade:

<!-- results: banking77/banking77-7b-8shot.md -->
| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, 8-shot, 4-bit | cuda | 0.783 | 0.779 | n/a | 264.0 | 7434.4 | 4.3 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.729 ± 0.011 | 0.697 ± 0.015 | 0.800 ± 0.003 | 4.4 | 89.3 | 358.5 | 91 | 60.3x |
| student, ONNX fp32 | cpu | 0.727 | 0.696 | 0.801 | 3.4 | 90.9 | 352.2 | 91 | 77.6x |
| student, ONNX int8 | cpu | 0.726 | 0.694 | 0.797 | 1.4 | 40.2 | 795.7 | 23 | 186.3x |
| baseline: student on 8 gold labels per class, no teacher | cpu | 0.490 ± 0.014 | 0.449 ± 0.018 | 0.473 ± 0.015 | n/a | n/a | n/a | n/a | n/a |
| upper bound: student on all gold labels | cpu | 0.914 ± 0.005 | 0.914 ± 0.005 | 0.775 ± 0.001 | n/a | n/a | n/a | n/a | n/a |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows 8 gold training examples per class (616 texts). The student learns from the teacher's soft labels on all 10003 training texts, plus cross-entropy on 8 gold labels per class (616 texts, weight 1.0). The baseline is the same student trained with cross-entropy on those 616 gold labels only, without the teacher. The upper bound is the same student trained with cross-entropy on all 10003 gold training labels and no teacher; its speed equals the student's. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 8-shot, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.080 | 0.726 | 0.694 | 1.4 |
| 10% | 0.231 | 0.744 | 0.718 | 27.8 |
| 20% | 0.302 | 0.750 | 0.730 | 54.2 |
| 30% | 0.355 | 0.764 | 0.748 | 80.6 |
| 50% | 0.440 | 0.778 | 0.769 | 133.4 |
| 100% | 1.000 | 0.783 | 0.779 | 265.4 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 8-shot, 4-bit on cuda.
<!-- /results -->

The distilled student is far better than the same student trained on the few gold labels alone, and runs on a
CPU in about a millisecond. Sending its least confident texts to the teacher closes most of the remaining gap
to the teacher.

Other students, the same teacher and soft labels. Students train for 5 epochs unless the name says otherwise.
`static` is the static-embedding student (`--student static`: token vectors from MiniLM-L6, averaged, no
attention), trained for 20 epochs at learning rate 1e-3:

<!-- results: banking77/students.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77-7b-8shot | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.729 ± 0.011 | 0.697 ± 0.015 | 0.800 ± 0.003 | ONNX int8 | 1.4 | 40.2 | 23 |
| all-MiniLM-L12-v2 | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `sentence-transformers/all-MiniLM-L12-v2` | 33 | 0.743 ± 0.011 | 0.710 ± 0.016 | 0.815 ± 0.004 | ONNX int8 | 3.3 | 79.2 | 34 |
| all-MiniLM-L12-v2-20-epochs | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `sentence-transformers/all-MiniLM-L12-v2` | 33 | 0.817 ± 0.001 | 0.814 ± 0.001 | 0.875 ± 0.002 | ONNX int8 | 3.2 | 82.0 | 34 |
| all-MiniLM-L6-v2-10-epochs | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.808 ± 0.003 | 0.801 ± 0.002 | 0.860 ± 0.004 | ONNX int8 | 1.1 | 38.1 | 23 |
| all-MiniLM-L6-v2-20-epochs | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.813 ± 0.002 | 0.811 ± 0.003 | 0.868 ± 0.003 | ONNX int8 | 1.2 | 36.3 | 23 |
| all-MiniLM-L6-v2-best-of-20 | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.807 ± 0.009 | 0.802 ± 0.010 | 0.862 ± 0.006 | ONNX int8 | 1.2 | 38.0 | 23 |
| all-mpnet-base-v2 | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `sentence-transformers/all-mpnet-base-v2` | 110 | 0.814 ± 0.004 | 0.805 ± 0.007 | 0.863 ± 0.001 | ONNX int8 | 7.3 | 220.4 | 110 |
| bge-small-en-v1.5 | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `BAAI/bge-small-en-v1.5` | 33 | 0.739 ± 0.008 | 0.706 ± 0.011 | 0.815 ± 0.004 | ONNX int8 | 2.7 | 75.2 | 34 |
| ModernBERT-base-20-epochs | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `answerdotai/ModernBERT-base` | 150 | 0.815 ± 0.002 | 0.812 ± 0.002 | 0.881 ± 0.001 | ONNX int8 | 9.7 | 262.8 | 151 |
| static | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.783 | `static` | 8 | 0.803 ± 0.001 | 0.801 ± 0.001 | 0.824 ± 0.001 | ONNX int8 | 0.1 | 5.1 | 8 |

All latencies on the CPU with 2 threads.
<!-- /results -->

Training longer matters more than a bigger student: with 20 epochs instead of 5, the 23M-parameter MiniLM-L6
student is as accurate as the 110M-parameter mpnet, at a sixth of its latency, and more accurate than its own
7B teacher. Learning from the teacher's soft labels on 10,000 texts, plus the 616 gold labels, smooths out
some of the teacher's individual mistakes. The static student is about a point behind the MiniLM students
trained as long, at a tenth of a millisecond per text and 8 MB. ModernBERT-base (150M parameters) reached
0.815, the same as MiniLM-L6 at about eight times its latency: its strength, inputs of up to 8,192 tokens, does
not matter for Banking77's short messages. Ten epochs get most of the way (MiniLM-L6:
0.808), so 10 is the default from 0.3.1. On the easier financial-tweets task, 20 epochs give the same accuracy
as 5 ([`financial-tweets-7b-20-epochs.md`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/financial-tweets/financial-tweets-7b-20-epochs.md)).
The other tables on this page were trained for 5 epochs, the default until 0.3.1. Choosing the epoch on 5%
held-out training texts (`--val-share 0.05`, up to 20 epochs; the `best-of-20` row) kept epochs 9, 11 and 20
for the three seeds and lands where 10 fixed epochs do: it finds a reasonable stopping point without gold
labels, but here it does not beat simply training longer.

### The most similar examples instead of 8 per class

[`examples/banking77/similar.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/banking77/similar.sh)
shows the 7B teacher, for each text, the 16 of the same 616 gold examples that are most similar to it
(`--few-shot-similar 16`) instead of all 616. The student trains for 10 epochs:

<!-- results: banking77/banking77-7b-similar16.md -->
| Model | Device | Accuracy | Macro-F1 | Top-3 accuracy | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, 16 similar examples, 4-bit | cuda | 0.851 | 0.850 | 0.947 | n/a | 156.8 | 4907.8 | 6.5 | 5443 | 1.0x |
| student, PyTorch | cpu | 0.857 ± 0.001 | 0.856 ± 0.001 | 0.947 ± 0.002 | 0.878 ± 0.003 | 3.7 | 72.7 | 440.4 | 91 | 42.6x |
| student, ONNX fp32 | cpu | 0.858 | 0.857 | 0.947 | 0.881 | 2.6 | 72.0 | 444.5 | 91 | 60.8x |
| student, ONNX int8 | cpu | 0.855 | 0.854 | 0.951 | 0.875 | 1.0 | 34.1 | 939.6 | 23 | 153.6x |

Task `banking77`, 3080 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The teacher's prompt shows each text the 16 most similar of 616 gold training examples. The student learns from the teacher's soft labels on all 10003 training texts, plus cross-entropy on 616 gold-labelled texts (8 per class; weight 1.0). Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, 16 similar examples, 4-bit on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.138 | 0.855 | 0.854 | 1.0 |
| 10% | 0.622 | 0.862 | 0.861 | 16.7 |
| 20% | 0.785 | 0.862 | 0.860 | 32.4 |
| 30% | 0.875 | 0.858 | 0.857 | 48.1 |
| 50% | 0.927 | 0.855 | 0.853 | 79.4 |
| 100% | 1.000 | 0.851 | 0.850 | 157.8 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, 16 similar examples, 4-bit on cuda.
<!-- /results -->

The teacher is more accurate with 16 well-chosen examples than with all 616 (0.851 against 0.783), and faster,
because its prompt is far shorter (157 ms per text against 264). The student follows: 0.855 deployed, above
its teacher and above the 0.813 of the same student trained for 20 epochs on the 8-per-class teacher. Its
cascade gains little, since the student is already about as accurate as the teacher it would ask.

[`examples/banking77/qwen35-similar.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/banking77/qwen35-similar.sh)
does the same with Qwen3.5-9B as the teacher:

<!-- results: banking77/banking77-qwen35-9b-similar16.md -->
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
<!-- /results -->

The newer teacher adds a point (0.861 against 0.851) and so does its student (0.867 in PyTorch, 0.859 deployed
in int8). It is far slower: Qwen3.5 cannot pack texts, so it labels about one text per second instead of six.

### Labelling on a budget

[`examples/banking77/budget.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/banking77/budget.sh)
has the teacher label only part of the training texts (`run --budget N`), chosen at random or over four rounds by
the student's uncertainty (`--rounds 4`). The teacher's labels come from the score cache of the run above, so only
the choice of texts differs:

<!-- results: banking77/budget.md -->
| Texts the teacher labels | At random (`--budget N`) | By the student's uncertainty (`--rounds 4`) |
|---|---|---|
| 1,000 of 10,003 | 0.480 ± 0.017 | 0.383 ± 0.016 |
| 2,500 of 10,003 | 0.735 ± 0.011 | 0.725 ± 0.017 |
| 5,000 of 10,003 | 0.830 ± 0.012 | 0.826 ± 0.011 |
| all 10,003 | 0.857 ± 0.001 | |

Banking77, 3080 test texts; the teacher of similar.sh (16 most similar examples), its labels read from the score cache. Student accuracy (PyTorch), mean ± standard deviation over 3 training seeds.
<!-- /results -->

A random half of the texts gets within three points of labelling them all. Choosing by the student's uncertainty
does worse than chance at every budget, by about ten points with 1,000 texts: the first students, trained on a
few hundred texts, pick ambiguous and unusual texts rather than useful ones. A random sample is therefore the
default.

### A contrastive warm-up of the student

[`examples/banking77/warmup.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/banking77/warmup.sh)
first trains the student's encoder on 20,000 SetFit-style pairs (`train --warmup-pairs 20000`: same class or
not, by gold label or the teacher's confident top label), then distils as usual. The teacher's labels come from
the score cache of the run above:

<!-- results: banking77/warmup.md -->
| Texts the teacher labels | Without warm-up | With `--warmup-pairs 20000` |
|---|---|---|
| 1,000 of 10,003, at random | 0.480 ± 0.017 | 0.510 ± 0.023 |
| 2,500 of 10,003, at random | 0.735 ± 0.011 | 0.739 ± 0.010 |
| all 10,003 | 0.857 ± 0.001 | 0.855 ± 0.002 |
| all 10,003, deployed (ONNX int8) | 0.855 | 0.850 |

Banking77, 3080 test texts; the teacher of similar.sh (16 most similar examples), its labels read from the score cache. Student accuracy (PyTorch unless marked), mean ± standard deviation over 3 training seeds.
<!-- /results -->

The warm-up helps only when the teacher labels few texts: three points with 1,000, nothing measurable with 2,500,
and nothing with all of them. It stays off by default.

### Rounds of teacher-written texts for the student's confusions

[`examples/banking77/improve.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/banking77/improve.sh)
runs `tinydistill improve` on the run above: three rounds in which the 7B teacher writes about 300 texts for the
label pairs the student confuses most on 1,000 held-out training texts, labels them, and the student is trained
again:

<!-- results: banking77/improve.md -->
| round | texts added | training texts | held-out agreement | test accuracy |
|---|---|---|---|---|
| 0 | 0 | 9003 | 0.854 | 0.856 |
| 1 | 294 | 9297 | 0.857 | 0.855 |
| 2 | 297 | 9594 | 0.863 | 0.853 |
| 3 | 296 | 9890 | 0.867 | 0.856 |

Held-out agreement: the student's agreement with the teacher on 1000 held-out training texts. Best by it: round 3 (runs/banking77-7b-similar16-improve/round-3). Round 0 is the run retrained without the held-out texts.
<!-- /results -->

The student agrees more with its teacher on the held-out texts after each round (0.854 to 0.867), but its test
accuracy does not move (0.856 before and after). On a task where the student already matches its teacher, the
extra texts teach it the teacher's view of the confusing pairs, not the gold labels.

### Compared with SetFit

[SetFit](https://github.com/huggingface/setfit) is a strong few-shot method that needs no LLM: it fine-tunes a
sentence encoder on pairs of labelled texts and fits a classifier head on top. Given the same 616 gold labels
([`examples/banking77/setfit_baseline.py`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/banking77/setfit_baseline.py)):

<!-- results: banking77/setfit.md -->
| Method | Gold labels | Also uses | Accuracy | Macro-F1 |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct, 8 examples per class in the prompt | 616 | | 0.783 | 0.779 |
| SetFit, MiniLM-L6 | 616 | | 0.797 ± 0.001 | 0.793 ± 0.002 |
| SetFit, mpnet-base | 616 | | 0.789 ± 0.005 | 0.788 ± 0.005 |
| tinydistill, MiniLM-L6 student | 616 | the 7B teacher's soft labels on 10,003 texts | 0.813 ± 0.002 | 0.811 ± 0.003 |
| tinydistill, static student | 616 | the same | 0.803 ± 0.001 | 0.801 ± 0.001 |
| MiniLM-L6 trained on all gold labels (upper bound) | 10,003 | | 0.914 ± 0.005 | 0.914 ± 0.005 |

Banking77, 3080 test texts; mean ± standard deviation over three seeds. Every method sees the same 616 gold-labelled texts (8 per class). SetFit: one epoch of contrastive training with 20 pair iterations, then a logistic-regression head (the library's defaults otherwise). The tinydistill students train for 20 epochs; the upper bound for 5.
<!-- /results -->

With the same encoder, distillation is about 1.6 points more accurate than SetFit. That margin comes from the
7B teacher and 10,000 unlabelled texts, and it costs a GPU for labelling; SetFit trains in about a minute and
is itself more accurate than the 7B teacher here. With few labels and no GPU, SetFit is a good choice; with
unlabelled texts and an LLM that already does the task well, distillation gets more out of them.

## CLINC150: many labels

Requests to a voice assistant: 150 intents plus "out of scope"
([CLINC150](https://huggingface.co/datasets/clinc/clinc_oos), CC BY 3.0), reproduced by
[`examples/clinc150/run.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/clinc150/run.sh). With 151 labels, scoring every label for every
text is the expensive part, so the teacher labels the data three ways:

<!-- results: clinc150/shortlist.md -->
| Teacher scores | Labelling time | Teacher accuracy | In scope | Out of scope | Gold label scored | Teacher latency (GPU) | Student accuracy (int8, CPU) |
|---|---|---|---|---|---|---|---|
| every label (151) | 95 min | 0.564 | 0.666 | 0.107 | 100% | 262 ms | 0.559 |
| `--shortlist 20` | 15 min | 0.555 | 0.674 | 0.020 | 80.3% | 73 ms | 0.555 |
| `--shortlist 20 --shortlist-keep oos` | 15 min | 0.581 | 0.672 | 0.175 | 96.8% | 87 ms | 0.574 |

CLINC150 (plus), 5,500 test texts of which 1,000 are out of scope; `Qwen/Qwen2.5-7B-Instruct` in 4 bit, zero-shot, labelling about 20,748 training and test texts on one GPU. Gold label scored: the share of test texts whose gold label the teacher scored at all.
<!-- /results -->

`--shortlist 20` labels six times faster and loses little on in-scope requests: their gold label is among the
20 most similar labels for 96% of them. But "out of scope" resembles no particular request, so it is almost
never shortlisted, and the teacher can no longer choose it. Scoring it for every text as well
(`--shortlist-keep oos`) fixes that, and is then more accurate than scoring every label, at the same speed as
the plain shortlist: with fewer competing labels, the teacher picks "out of scope" more often when it is right.
A zero-shot teacher still recognises few out-of-scope requests; the same student trained on all gold labels
reaches 0.87 ([`clinc150-7b-full.md`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/clinc150/clinc150-7b-full.md)).

### "None of these" as an outside label, and similar examples

[`examples/clinc150/outside.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/clinc150/outside.sh)
makes out of scope the task's `outside` label (`outside: none of these`, listed last in the prompt) instead of one
intent among the others, with the same shortlist of 20 plus that label. Then the teacher also sees each text's
16 most similar of 8 gold examples per label (`--few-shot-similar 16 --gold-per-class 8`):

<!-- results: clinc150/outside.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clinc150-7b-shortlist20-keep-oos | zero-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.581 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.573 ± 0.003 | 0.604 ± 0.005 | 0.724 ± 0.003 | ONNX int8 | 1.6 | 29.0 | 23 |
| clinc150-7b-outside | zero-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.575 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.567 ± 0.007 | 0.594 ± 0.004 | 0.719 ± 0.005 | ONNX int8 | 1.0 | 22.7 | 23 |
| clinc150-7b-outside-similar16 | 16 similar examples, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.769 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.767 ± 0.002 | 0.832 ± 0.001 | 0.791 ± 0.001 | ONNX int8 | 1.6 | 42.5 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

The outside label alone changes little (0.567 against 0.573). The similar examples are what matter: the teacher
rises from 0.575 to 0.769 and the student follows to 0.767, with 0.915 in-scope accuracy deployed
([`clinc150-7b-outside-similar16.md`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/clinc150/clinc150-7b-outside-similar16.md)).
Out-of-scope requests are still the weak part: the teacher marks few of them as such (recall 0.126), though
nearly always correctly (precision 0.969).

### Scored as the few-shot papers score it: 150 intents

The few-shot papers train and test on the 150 intents alone, where these runs have out of scope as a 151st
label and 5,500 test texts.
[`examples/clinc150/in_scope.py`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/clinc150/in_scope.py)
scores two saved Gemma 4 12B runs (described [below](#the-probe-alone-minutes-of-labelling-instead-of-hours))
on the 4,500 in-scope test texts, taking the best of the 150 intents as a model trained without out of scope
would answer ([`in-scope.json`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/clinc150/in-scope.json)):

- the recommended settings, `--probe-only` with 16 similar examples (37 minutes of labelling): the deployed
  int8 student scores 0.951 (0.950 when its "out of scope" answers count as wrong), the PyTorch students
  0.952 ± 0.002, the teacher 0.967;
- the most accurate run on all 5,500 texts, the in-context teacher plus the probe (7.2 hours): the deployed
  student 0.945 (0.943), the PyTorch students 0.948 ± 0.002, the teacher 0.964.

So the probe-only run is the better of the two on the 150 intents, and loses its 0.008 overall on the
out-of-scope texts. Published results, reported in the papers and not rerun, use 5 or 10 gold labels per class
where these runs use 8, and train on in-scope texts only. With 5 and 10: CPFT 0.9234 and 0.9418 (Zhang et al.
2021, Table 2, [arXiv 2109.06349](https://arxiv.org/abs/2109.06349)); SetFit 0.8868 and 0.9067, DeBERTa 0.9186
and 0.9505, LLaMA-2 7B classifying every test text with 20 retrieved examples 0.9413 and 0.952 (Milios et al.
2023, Table 1, [arXiv 2309.10954](https://arxiv.org/abs/2309.10954)). The deployed student's 0.951 with 8 is
above CPFT and SetFit even at 10, level with DeBERTa and LLaMA-2 7B at 10, and below the longer prompts and
larger LLMs there, each classifying every test text: LLaMA-2 7B with a 4K prompt 0.9535 and 0.9602, LLaMA-2
70B 0.9642 and 0.9713, and 0.9756 and 0.9818 with a 4K prompt. The papers do not say which 4,500 test texts
they use beyond the 150 intents, so take the comparison as a rough one.

## AG News: the task file depends on the teacher

News articles in four topics ([AG News](https://huggingface.co/datasets/fancyzhx/ag_news)), the example in the
"your own task" notebook. With short descriptions of the labels in the prompt, `Qwen/Qwen2.5-1.5B-Instruct`
answered "world" for nearly every article: test accuracy 0.304 on four balanced topics
([`ag-news-1.5b-with-descriptions.md`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/ag-news/ag-news-1.5b-with-descriptions.md)).
`tinydistill tune` compares versions of the task file on 200 gold-labelled training texts before anything is
labelled:

<!-- results: ag-news/tune-1.5b.md -->
| Teacher | Prompt | Few-shot examples per class | Dev accuracy | Dev macro-F1 | Seconds per text |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | task.yaml | 0 | 0.805 | 0.806 | 0.007 |
| `Qwen/Qwen2.5-1.5B-Instruct` | narrow-world.yaml | 0 | 0.315 | 0.246 | 0.005 |
| `Qwen/Qwen2.5-1.5B-Instruct` | with-descriptions.yaml | 0 | 0.305 | 0.229 | 0.005 |

Scored on 200 gold-labelled training texts held out from the few-shot examples (not the test set), best first.

Best: `tinydistill run examples/ag-news/task.yaml --teacher Qwen/Qwen2.5-1.5B-Instruct --data ... --out ...`
<!-- /results -->

The 7B teacher prefers the opposite, narrower descriptions:

<!-- results: ag-news/tune-7b.md -->
| Teacher | Prompt | Few-shot examples per class | Dev accuracy | Dev macro-F1 | Seconds per text |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` (4-bit) | narrow-world.yaml | 0 | 0.785 | 0.804 | 0.016 |
| `Qwen/Qwen2.5-7B-Instruct` (4-bit) | task.yaml | 0 | 0.640 | 0.655 | 0.018 |

Scored on the same 200 gold-labelled training texts as tune-1.5b.md, best first.
<!-- /results -->

So the same task file can suit one teacher and fail another, and a check on a few hundred labelled texts
finds out before hours of labelling; the notebook now does this and uses the winner. With the task file
`tune` chose, on the full test set (7,600 articles):

<!-- results: ag-news/ag-news-1.5b.md -->
| Model | Device | Accuracy | Macro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot | cuda | 0.749 | 0.752 | n/a | 32.1 | 141.8 | 225.7 | 3087 | 1.0x |
| student, PyTorch | cpu | 0.775 ± 0.009 | 0.777 ± 0.009 | 0.830 ± 0.004 | 9.2 | 374.7 | 85.4 | 91 | 3.5x |
| student, ONNX fp32 | cpu | 0.785 | 0.788 | 0.831 | 7.8 | 449.6 | 71.2 | 91 | 4.1x |
| student, ONNX int8 | cpu | 0.781 | 0.783 | 0.833 | 3.7 | 268.9 | 119.0 | 23 | 8.6x |
| upper bound: student on all gold labels | cpu | 0.896 ± 0.003 | 0.895 ± 0.002 | 0.741 ± 0.007 | n/a | n/a | n/a | n/a | n/a |

Task `ag-news`, 7600 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns only from the teacher's soft labels on 10000 unlabelled training texts. The upper bound is the same student trained with cross-entropy on all 10000 gold training labels and no teacher; its speed equals the student's. Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 10 epochs, stopping after 3 without improvement); kept epochs: 2, 3, 7, 8. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot on cuda.

### Cascade: student first, unsure texts to the teacher

| Share sent to the teacher | Student confidence threshold | Accuracy | Macro-F1 | Mean latency per text (ms) |
|---|---|---|---|---|
| 0% | 0.269 | 0.781 | 0.783 | 3.7 |
| 10% | 0.459 | 0.781 | 0.783 | 6.9 |
| 20% | 0.517 | 0.776 | 0.779 | 10.2 |
| 30% | 0.570 | 0.772 | 0.774 | 13.4 |
| 50% | 0.678 | 0.758 | 0.760 | 19.8 |
| 100% | 1.000 | 0.749 | 0.752 | 35.9 |

The student answers every text and passes its least confident ones to the teacher. Thresholds are set on the test texts to hit each share; in deployment, set them on a validation split. Latency is the student's batch-1 latency plus the share times the batch-1 latency of the teacher, zero-shot on cuda.
<!-- /results -->

The teacher goes from 0.304 to 0.749, and the student, trained on its soft labels for 10,000 articles, is a
little more accurate than the teacher.

### A reasoning step before the answer

[`examples/ag-news/reason.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/ag-news/reason.sh)
labels the same 3,000 training and 2,000 test articles with the 1.5B teacher twice: scoring the labels directly,
and after the teacher writes up to 256 tokens of reasoning (`label --reason`):

<!-- results: ag-news/reason.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ag-news-1.5b-plain3k | zero-shot `Qwen/Qwen2.5-1.5B-Instruct` | 0.755 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.785 ± 0.002 | 0.785 ± 0.002 | 0.825 ± 0.010 | ONNX int8 | 3.4 | 208.9 | 23 |
| ag-news-1.5b-reason3k | zero-shot, reasoning `Qwen/Qwen2.5-1.5B-Instruct` | 0.778 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.785 ± 0.004 | 0.777 ± 0.004 | 0.914 ± 0.003 | ONNX int8 | 3.5 | 208.3 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

Reasoning makes the teacher more accurate (0.778 against 0.755) and its student agrees with it more, but the
student is not more accurate (0.785 both ways, with a slightly lower macro-F1), so here the extra labelling time
buys nothing for the deployed model.

### The teacher rewrites its own instruction

[`examples/ag-news/propose.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/ag-news/propose.sh)
has the 1.5B teacher rewrite the instruction before the prompt's first placeholder (`tune --propose 5 --rounds 3`,
shown its mistakes on 100 dev texts and scored on 100 others;
[`propose-1.5b.md`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/ag-news/propose-1.5b.md)).
Its best rewrite, "What is this news headline primarily about?", scored 0.880 on the dev texts against 0.850. The
best task file is then used alone and with the original as a prompt ensemble, with the settings of the first
run above:

<!-- results: ag-news/propose-runs.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ag-news-1.5b | zero-shot `Qwen/Qwen2.5-1.5B-Instruct` | 0.749 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.775 ± 0.009 | 0.777 ± 0.009 | 0.830 ± 0.004 | ONNX int8 | 3.7 | 268.9 | 23 |
| ag-news-1.5b-proposed | zero-shot `Qwen/Qwen2.5-1.5B-Instruct` | 0.778 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.803 ± 0.002 | 0.806 ± 0.002 | 0.863 ± 0.002 | ONNX int8 | 4.2 | 309.6 | 23 |
| ag-news-1.5b-ensemble | zero-shot, 2 prompts `Qwen/Qwen2.5-1.5B-Instruct` | 0.770 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.796 ± 0.005 | 0.798 ± 0.005 | 0.852 ± 0.001 | ONNX int8 | 3.8 | 308.8 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

The rewritten instruction lifts the teacher by three points on the test set (0.778 against 0.749) and the
student by about as much (0.803 against 0.775). Adding the original prompt as a second member gains nothing over
the rewrite alone. Most rewrites were worse than the original on the dev texts, so the dev score, not the
teacher's choice, picks the prompt.

## Financial tweets: an easier task

Sentiment of financial news tweets, 3 classes
([Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment),
MIT licence per the dataset card), reproduced by
[`examples/financial-tweets/run.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/financial-tweets/run.sh).
The validation split is the test set. The rows include an ensemble of three model families (Qwen2.5-7B,
Mistral-7B and Phi-3.5-mini, all 4 bit) and a student trained only on texts the teacher wrote itself
(`tinydistill synth`, no real training texts):

<!-- results: financial-tweets/teachers.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| financial-tweets-1.5b | zero-shot `Qwen/Qwen2.5-1.5B-Instruct` | 0.537 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.514 ± 0.004 | 0.538 ± 0.004 | 0.810 ± 0.004 | ONNX int8 | 1.6 | 85.1 | 23 |
| financial-tweets-1.5b-8shot | 8-shot `Qwen/Qwen2.5-1.5B-Instruct` | 0.314 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.295 ± 0.003 | 0.276 ± 0.003 | 0.870 ± 0.006 | ONNX int8 | 1.8 | 77.2 | 23 |
| financial-tweets-7b | zero-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.832 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.825 ± 0.007 | 0.768 ± 0.009 | 0.870 ± 0.005 | ONNX int8 | 2.5 | 105.7 | 23 |
| financial-tweets-ensemble | teacher ensemble of 3, zero-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct,mistralai/Mistral-7B-Instruct-v0.3,microsoft/Phi-3.5-mini-instruct` | 0.842 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.828 ± 0.006 | 0.778 ± 0.005 | 0.884 ± 0.007 | ONNX int8 | 1.6 | 99.4 | 23 |
| financial-tweets-synthetic | zero-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.832 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.596 ± 0.037 | 0.571 ± 0.027 | 0.615 ± 0.045 | ONNX int8 | 1.6 | 110.4 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

Here the 7B teacher is close to the student trained on all gold labels, and the student is close to the
teacher. The ensemble adds a little. Few-shot examples hurt on this skewed task (see
[Limitations](#limitations)). The student that saw only about 1,500 texts the teacher wrote itself is far
behind the one trained on real tweets: generated texts are much less varied than real ones, so `synth` is a
way to start without data, not a replacement for it. Full tables per run are in
[`examples/financial-tweets/`](https://github.com/Andreas-Hum/tinydistill/tree/main/examples/financial-tweets).

## GoEmotions: multi-label

Reddit comments with any of 27 emotions or neutral
([GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions), Apache 2.0), reproduced
by [`examples/go-emotions/run.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/go-emotions/run.sh).
The teacher answers one yes/no question per label; accuracy is the share of texts with exactly the right set
of labels.

<!-- results: go-emotions/go-emotions-1.5b.md -->
| Model | Device | Accuracy | Macro-F1 | Micro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot | cpu | 0.004 | 0.191 | 0.186 | n/a | 9990.9 | n/a | 0.1 | 6175 | 1.0x |
| teacher, zero-shot | cuda | 0.004 | 0.191 | 0.186 | n/a | 74.3 | 2131.6 | 15.0 | 3087 | 134.5x |
| student, PyTorch | cpu | 0.002 ± 0.001 | 0.111 ± 0.004 | 0.141 ± 0.002 | 0.015 ± 0.002 | 5.4 | 116.3 | 275.2 | 91 | 1857.3x |
| student, ONNX fp32 | cpu | 0.002 | 0.106 | 0.141 | 0.016 | 4.8 | 115.7 | 276.6 | 91 | 2072.9x |
| student, ONNX int8 | cpu | 0.002 | 0.108 | 0.141 | 0.016 | 2.0 | 66.3 | 482.8 | 23 | 5026.0x |
| baseline: student on 8 gold labels per class, no teacher | cpu | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.007 ± 0.000 | n/a | n/a | n/a | n/a | n/a |
| upper bound: student on all gold labels | cpu | 0.225 ± 0.021 | 0.034 ± 0.019 | 0.320 ± 0.034 | 0.007 ± 0.000 | n/a | n/a | n/a | n/a | n/a |

Task `go-emotions`, 5427 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-1.5B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns from the teacher's soft labels on all 10000 training texts, plus cross-entropy on 8 gold labels per class (223 texts, weight 1.0). The baseline is the same student trained with cross-entropy on those 223 gold labels only, without the teacher. The upper bound is the same student trained with cross-entropy on all 10000 gold training labels and no teacher; its speed equals the student's. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency. Speed-up is relative to the teacher, zero-shot on cpu.
<!-- /results -->

This is a hard setting for a 1.5B teacher: it answers yes far too often, so exact-set accuracy is near zero,
and the student inherits that.

With `Qwen/Qwen2.5-7B-Instruct` (4 bit) the teacher is better, and with enough training its student nearly
matches it. Answering 28 yes/no questions per text takes the student longer to learn than one choice among
labels: with 10 epochs every seed kept its last epoch and the student predicted too few labels
([`go-emotions-7b.md`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/go-emotions/go-emotions-7b.md)). With up to 30 epochs and the epoch
chosen on held-out texts (`--epochs 30 --val-share 0.05`):

<!-- results: go-emotions/go-emotions-7b-30-epochs.md -->
| Model | Device | Accuracy | Macro-F1 | Micro-F1 | Agreement with teacher | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Throughput (texts/s) | Size (MB) | Speed-up, batch 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher, zero-shot, 4-bit | cuda:0 | 0.045 | 0.277 | 0.268 | n/a | n/a | n/a | n/a | n/a | n/a |
| student, PyTorch | cpu | 0.057 ± 0.005 | 0.226 ± 0.017 | 0.253 ± 0.003 | 0.157 ± 0.005 | 5.9 | 107.4 | 297.9 | 91 | n/a |
| student, ONNX fp32 | cpu | 0.060 | 0.208 | 0.250 | 0.161 | 3.7 | 103.2 | 310.0 | 91 | n/a |
| student, ONNX int8 | cpu | 0.059 | 0.199 | 0.246 | 0.160 | 1.6 | 50.1 | 639.3 | 23 | n/a |
| baseline: student on 8 gold labels per class, no teacher | cpu | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.128 ± 0.000 | n/a | n/a | n/a | n/a | n/a |
| upper bound: student on all gold labels | cpu | 0.308 ± 0.009 | 0.134 ± 0.003 | 0.431 ± 0.011 | 0.063 ± 0.002 | n/a | n/a | n/a | n/a | n/a |

Task `go-emotions`, 5427 gold-labelled test texts. Teacher: `Qwen/Qwen2.5-7B-Instruct`. Student: `sentence-transformers/all-MiniLM-L6-v2`. The student learns from the teacher's soft labels on all 10000 training texts, plus cross-entropy on 223 gold-labelled texts (8 per class; weight 1.0). The baseline is the same student trained with cross-entropy on those 223 gold labels only, without the teacher. The upper bound is the same student trained with cross-entropy on all 10000 gold training labels and no teacher; its speed equals the student's. Each student holds out 5% of its training texts and keeps the epoch with the lowest loss on them (up to 30 epochs, stopping after 3 without improvement); kept epochs: 9, 10, 11, 12, 15, 16, 18, 22. Values with ± are the mean and standard deviation over 3 training seeds; the ONNX rows are the exported first-seed model.
CPU rows use 2 threads. Latency is the median over repeated runs. Size is the teacher's weights as loaded and the students' model files. The teacher packs several texts into one forward pass; it is timed at batch 32 only when that takes under three minutes, otherwise its throughput is 1 / batch-1 latency.
<!-- /results -->

The student is within about 0.05 macro-F1 of its teacher and more often gets the exact set of labels right.
Its macro-F1 is also higher than that of the same student trained on all 10,000 gold labels, which rarely
predicts the uncommon emotions; the gold-trained student is better on micro-F1, which the common labels
dominate. With only 8 gold labels per emotion and no teacher, the student predicts no labels at all.

## Danish MASSIVE: an English prompt over Danish texts

Intent detection in Danish with 60 intents ([MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_intent),
CC BY 4.0; 11,514 training and 2,974 test texts), reproduced by
[`examples/massive-da/run.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/massive-da/run.sh).
The task file's instruction and label names are in English; only the texts are Danish. The 8-shot runs also
train on those 8 gold labels per intent:

<!-- results: massive-da/summary.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| massive-da-7b | zero-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.603 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.594 ± 0.007 | 0.536 ± 0.008 | 0.725 ± 0.003 | ONNX int8 | 0.8 | 29.6 | 23 |
| massive-da-7b-8shot | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.736 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.724 ± 0.006 | 0.667 ± 0.004 | 0.752 ± 0.006 | ONNX int8 | 0.9 | 29.5 | 23 |
| paraphrase-multilingual-MiniLM-L12-v2 | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.736 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118 | 0.761 ± 0.002 | 0.707 ± 0.003 | 0.804 ± 0.006 | ONNX int8 | 1.9 | 39.3 | 118 |

All latencies on the CPU with 2 threads.
<!-- /results -->

The English teacher prompt works on Danish texts, and the 8 examples per intent add 13 points to it. The
English MiniLM student gets within about a point of its teacher; the multilingual
`paraphrase-multilingual-MiniLM-L12-v2` student is more accurate than the teacher (0.761 against 0.736), at
twice the latency and five times the size.

Most of the multilingual student's size is its vocabulary of 250,000 tokens, of which a Danish task uses few.
[`examples/massive-da/trim.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/massive-da/trim.sh)
trains the same student on the same labels with `train --trim-vocab`, which keeps only the tokens of the
training texts (and the 1,000 most frequent others):

<!-- results: massive-da/trim.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.736 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 118 | 0.761 ± 0.002 | 0.707 ± 0.003 | 0.804 ± 0.006 | ONNX int8 | 1.9 | 39.3 | 118 |
| paraphrase-multilingual-MiniLM-L12-v2-trimmed | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.736 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 24 | 0.761 ± 0.001 | 0.708 ± 0.003 | 0.802 ± 0.006 | ONNX int8 | 1.8 | 40.2 | 24 |

All latencies on the CPU with 2 threads.
<!-- /results -->

Trimmed, it is a fifth of the size (24 MB, as small as the English MiniLM-L6) at the same accuracy and latency.

## IMDB: long reviews

Film reviews are often several hundred tokens long, while the student reads 128
([IMDB](https://huggingface.co/datasets/stanfordnlp/imdb), Maas et al., 2011; 5,000 training and 2,000 test
reviews, shuffled), reproduced by
[`examples/imdb/run.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/imdb/run.sh). Either
the student reads the first 128 tokens of each review, or every review is cut into windows of 128 tokens, 64
apart (`--chunk-tokens 128 --chunk-stride 64`): the teacher scores each window, the student trains on the
windows, and at prediction time it averages its windows' probabilities:

<!-- results: imdb/summary.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| imdb-truncated | zero-shot `Qwen/Qwen2.5-1.5B-Instruct` | 0.934 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.844 ± 0.007 | 0.843 ± 0.007 | 0.851 ± 0.005 | ONNX int8 | 6.4 | 217.4 | 23 |
| imdb-chunked | zero-shot `Qwen/Qwen2.5-1.5B-Instruct` | 0.875 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.860 ± 0.006 | 0.858 ± 0.006 | 0.926 ± 0.001 | ONNX int8 | 18.0 | 866.0 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

Reading in windows makes the student 1.6 points more accurate (0.860 against 0.844) and much closer to its
teacher, at about three times the latency, since a review is read as several windows. The teacher is less
accurate on windows (0.875) than on whole reviews (0.934), because one window often says too little about the
whole review; the truncated student learns from the whole-review teacher yet still sees only the first 128
tokens.

## HWU64: a task the settings were not tuned on

Every setting on this page was chosen on the tasks above. HWU64 (64 intents; 8,954 training and 1,076 test
texts) is new to them: [`examples/hwu64/run.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/hwu64/run.sh)
takes the settings of [Which settings to use](https://andreas-hum.github.io/tinydistill/when-to-use/#which-settings-to-use)
for about 8 gold labels per class, with Gemma 4 12B as the teacher, and changes nothing else. Its task file
gives the intents' readable names and nothing more. With the probe alone (`--probe-only`), once with a zero-shot
prompt and once with each text's 16 most similar of the 512 gold examples, and then with the teacher scoring
every label with those examples, averaged with the probe:

<!-- results: hwu64/summary.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hwu64-gemma4-12b-probe-only-zeroshot | zero-shot, 4-bit, probe only `google/gemma-4-12B-it` | 0.842 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.865 ± 0.001 | 0.865 ± 0.004 | 0.880 ± 0.003 | ONNX int8 | 1.1 | 30.9 | 23 |
| hwu64-gemma4-12b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `google/gemma-4-12B-it` | 0.879 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.891 ± 0.004 | 0.889 ± 0.004 | 0.897 ± 0.004 | ONNX int8 | 1.5 | 30.3 | 23 |
| hwu64-gemma4-12b-similar16 | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.877 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.894 ± 0.005 | 0.893 ± 0.004 | 0.893 ± 0.003 | ONNX int8 | 0.8 | 21.7 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

The student is more accurate than its teacher in both: 0.865 against 0.842 with the zero-shot prompt, after
5.5 minutes of labelling, and 0.891 against 0.879 with the similar examples, after 16 minutes (0.888 deployed
in int8). The same student trained on the 512 gold labels alone reaches 0.639
([`hwu64-gemma4-12b-probe-only-zeroshot.json`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/hwu64/hwu64-gemma4-12b-probe-only-zeroshot.json),
`--gold-only`). The teacher scoring every label adds little for its cost: 7.7 hours of labelling for a
student of 0.894 (0.892 deployed), against 16 minutes for 0.891
([`hwu64-gemma4-12b-similar16.json`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/hwu64/hwu64-gemma4-12b-similar16.json)),
and its teacher is no more accurate (0.877 against 0.879). As on the tuned tasks, the probe alone is the
better trade. The 7.7-hour student is the
[HWU64 model on the Hub](https://huggingface.co/Shroomy/tinydistill-hwu64-minilm) and in the
[live demo](https://andreas-hum.github.io/tinydistill/demo/).

Published few-shot results on HWU64, reported in the papers and not rerun, use 5 or 10 gold labels per class
where this run uses 8, and larger models. With 5 and 10: CPFT 0.8203 and 0.8713 (Zhang et al. 2021, Table 2,
[arXiv 2109.06349](https://arxiv.org/abs/2109.06349)); SetFit 0.7838 and 0.8335, DeBERTa 0.7980 and 0.8693,
and LLaMA-2 7B classifying every test text with 20 retrieved examples 0.8755 and 0.8782 (Milios et al. 2023,
Table 1, [arXiv 2309.10954](https://arxiv.org/abs/2309.10954)). The student's 0.891 with 8 is above these; with
10 it is below the larger LLMs or longer prompts there, each classifying every test text: LLaMA 65B 0.9006,
LLaMA-2 7B with a 4K prompt 0.9033, LLaMA-2 70B 0.8977, and 0.9173 with a 4K prompt. The training set here has the same 8,954
texts as theirs, but whether these 1,076 test texts are their test set or their validation set (both have
1,076) is not stated by the dataset, so take the comparison as a rough one.

## Newer teachers: Qwen3.5 and Gemma 4

[`examples/teachers-2026/run.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/teachers-2026/run.sh)
scores the same dev splits that `tune` uses with Qwen2.5-7B and four newer models, all in 4 bit: teacher accuracy
only, before any student is trained. AG News (200 dev texts, zero-shot):

<!-- results: teachers-2026/ag-news.md -->
| Teacher | Prompt | Few-shot examples per class | Dev accuracy | Dev macro-F1 | Seconds per text |
|---|---|---|---|---|---|
| `Qwen/Qwen3.5-4B` | task.yaml | 0 | 0.820 | 0.840 | 0.179 |
| `Qwen/Qwen3.5-9B` | task.yaml | 0 | 0.795 | 0.814 | 0.213 |
| `google/gemma-4-E4B-it` | task.yaml | 0 | 0.765 | 0.784 | 0.182 |
| `google/gemma-4-12B-it` | task.yaml | 0 | 0.750 | 0.770 | 0.228 |
| `Qwen/Qwen2.5-7B-Instruct` | task.yaml | 0 | 0.640 | 0.655 | 0.015 |

Scored on 200 gold-labelled training texts held out from the few-shot examples (not the test set), best first.

Best: `tinydistill run examples/ag-news/task.yaml --teacher Qwen/Qwen3.5-4B --data ... --out ...`
<!-- /results -->

Banking77 (500 dev texts), zero-shot and with 8 examples per class:

<!-- results: teachers-2026/banking77.md -->
| Teacher | Prompt | Few-shot examples per class | Dev accuracy | Dev macro-F1 | Seconds per text |
|---|---|---|---|---|---|
| `Qwen/Qwen3.5-9B` | task.yaml | 8 | 0.850 | 0.836 | 6.652 |
| `google/gemma-4-12B-it` | task.yaml | 8 | 0.840 | 0.831 | 11.957 |
| `Qwen/Qwen3.5-4B` | task.yaml | 8 | 0.804 | 0.795 | 6.619 |
| `google/gemma-4-E4B-it` | task.yaml | 8 | 0.794 | 0.780 | 3.917 |
| `Qwen/Qwen2.5-7B-Instruct` | task.yaml | 8 | 0.768 | 0.744 | 0.187 |
| `google/gemma-4-12B-it` | task.yaml | 0 | 0.712 | 0.680 | 2.936 |
| `Qwen/Qwen3.5-9B` | task.yaml | 0 | 0.668 | 0.641 | 1.471 |
| `Qwen/Qwen3.5-4B` | task.yaml | 0 | 0.638 | 0.611 | 1.268 |
| `Qwen/Qwen2.5-7B-Instruct` | task.yaml | 0 | 0.616 | 0.571 | 0.115 |
| `google/gemma-4-E4B-it` | task.yaml | 0 | 0.572 | 0.541 | 0.674 |

Scored on 500 gold-labelled training texts held out from the few-shot examples (not the test set), best first.

Best: `tinydistill run examples/banking77/task.yaml --teacher Qwen/Qwen3.5-9B --few-shot 8 --data ... --out ...`
<!-- /results -->

Danish MASSIVE (500 dev texts), zero-shot and with 8 examples per intent:

<!-- results: teachers-2026/massive-da.md -->
| Teacher | Prompt | Few-shot examples per class | Dev accuracy | Dev macro-F1 | Seconds per text |
|---|---|---|---|---|---|
| `google/gemma-4-12B-it` | task.yaml | 8 | 0.814 | 0.779 | 10.780 |
| `Qwen/Qwen3.5-9B` | task.yaml | 8 | 0.798 | 0.772 | 6.159 |
| `Qwen/Qwen3.5-4B` | task.yaml | 8 | 0.780 | 0.709 | 6.020 |
| `google/gemma-4-E4B-it` | task.yaml | 8 | 0.752 | 0.740 | 3.662 |
| `google/gemma-4-12B-it` | task.yaml | 0 | 0.730 | 0.717 | 2.609 |
| `Qwen/Qwen2.5-7B-Instruct` | task.yaml | 8 | 0.700 | 0.701 | 0.180 |
| `Qwen/Qwen3.5-9B` | task.yaml | 0 | 0.696 | 0.670 | 1.459 |
| `google/gemma-4-E4B-it` | task.yaml | 0 | 0.690 | 0.680 | 0.728 |
| `Qwen/Qwen3.5-4B` | task.yaml | 0 | 0.688 | 0.665 | 1.422 |
| `Qwen/Qwen2.5-7B-Instruct` | task.yaml | 0 | 0.598 | 0.561 | 0.127 |

Scored on 500 gold-labelled training texts held out from the few-shot examples (not the test set), best first.

Best: `tinydistill run examples/massive-da/task.yaml --teacher google/gemma-4-12B-it --few-shot 8 --data ... --out ...`
<!-- /results -->

Every newer teacher is more accurate than Qwen2.5-7B with the same prompt: on AG News zero-shot the 4B Qwen3.5
reaches 0.820 against 0.640, with 8 examples per class on Banking77 Qwen3.5-9B reaches 0.850 against 0.768, and on
Danish Gemma 4 12B reaches 0.814 against 0.700. They were also much slower per text in these runs (4 to 12 seconds
against under 0.2 with 8 examples per class), because their linear-attention or sliding-window layers kept
tinydistill from packing texts and it scored them one at a time; later versions score them as rows of a batch.

## Several sets of similar examples: worse than one

[`examples/teachers-2026/sets.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/teachers-2026/sets.sh)
scores each text three times with the 7B teacher (`--few-shot-sets 3`): with its 16 most similar gold examples,
then with the next 16 and the 16 after those, and averages the three label distributions (demonstration
ensembling). Against the same teacher with one set of 16, on Banking77:

<!-- results: teachers-2026/sets/banking77.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77-7b-similar16 | 16 similar examples, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.851 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.857 ± 0.001 | 0.856 ± 0.001 | 0.878 ± 0.003 | ONNX int8 | 1.0 | 34.1 | 23 |
| banking77-7b-similar16-sets3 | 16 similar examples, 3 example sets, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.749 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.809 ± 0.002 | 0.801 ± 0.002 | 0.795 ± 0.003 | ONNX int8 | 0.9 | 34.0 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

and on CLINC150, with the shortlist of 20 plus out of scope:

<!-- results: teachers-2026/sets/clinc150.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clinc150-7b-similar16 | 16 similar examples, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.775 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.767 ± 0.006 | 0.834 ± 0.003 | 0.796 ± 0.004 | ONNX int8 | 0.8 | 22.2 | 23 |
| clinc150-7b-similar16-sets3 | 16 similar examples, 3 example sets, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.702 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.749 ± 0.005 | 0.816 ± 0.003 | 0.734 ± 0.004 | ONNX int8 | 0.9 | 23.0 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

Three sets lower the teacher by 10 points on Banking77 (0.851 to 0.749) and 7 on CLINC150 (0.775 to 0.702), and
the students follow (0.857 to 0.809, 0.767 to 0.749), at three times the labelling cost. With 8 gold examples
per class, a text's class has at most 8 examples to show, so the later sets hold few or none of them, and
averaging those sets in dilutes the one that does. One set stays the default.

## A linear probe on the teacher's hidden states

[`examples/teachers-2026/probe.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/teachers-2026/probe.sh)
adds `--probe` to the 7B teacher with each text's 16 most similar gold examples: a linear classifier on the
teacher's last hidden state for the prompt plus the text, fit on the same 8 gold examples per class, whose
probabilities are averaged with the in-context ones. On Banking77:

<!-- results: teachers-2026/probe/banking77.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77-7b-similar16 | 16 similar examples, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.851 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.857 ± 0.001 | 0.856 ± 0.001 | 0.878 ± 0.003 | ONNX int8 | 1.0 | 34.1 | 23 |
| banking77-7b-similar16-probe | 16 similar examples, 4-bit, probe `Qwen/Qwen2.5-7B-Instruct` | 0.859 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.875 ± 0.001 | 0.875 ± 0.001 | 0.890 ± 0.001 | ONNX int8 | 1.0 | 33.7 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

and on CLINC150, with the shortlist of 20 plus out of scope:

<!-- results: teachers-2026/probe/clinc150.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clinc150-7b-similar16 | 16 similar examples, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.775 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.767 ± 0.006 | 0.834 ± 0.003 | 0.796 ± 0.004 | ONNX int8 | 0.8 | 22.2 | 23 |
| clinc150-7b-similar16-probe | 16 similar examples, 4-bit, probe `Qwen/Qwen2.5-7B-Instruct` | 0.842 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.817 ± 0.007 | 0.873 ± 0.004 | 0.832 ± 0.007 | ONNX int8 | 0.9 | 22.1 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

The probe lifts the teacher from 0.851 to 0.859 on Banking77 and from 0.775 to 0.842 on CLINC150, and the
students gain more than their teachers: 0.857 to 0.875 and 0.767 to 0.817, or 0.872 and 0.813 deployed in int8
(from 0.855 and 0.760). The Banking77 student is the most accurate on this page with 8 gold labels per class (level with the
student of the two-teacher ensemble below, 0.875, and above it in int8, 0.872 against 0.870), above the student
of Qwen3.5-9B (0.867), with a teacher that takes 155 ms per text on the GPU against Qwen3.5-9B's
924 ms. On the gold examples held out
from the fit (`--probe-dev`), the probe alone is below the prompt on Banking77 (75.6% against 78.0% of 123) and
above it on CLINC150 (96.7% against 94.6% of 242), and combined beats both on Banking77 (78.9%) but not the
probe alone on CLINC150 (95.5%); those held-out texts are few, and the test sets are the measure.
The Banking77 student is on the [Hub](https://huggingface.co/Shroomy/tinydistill-banking77-minilm) and in the
[live demo](https://andreas-hum.github.io/tinydistill/demo/); on CLINC150 the student of Gemma 4 12B (below)
replaced this one.

## The newer teachers with similar examples, and both together

[`examples/teachers-2026/similar.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/teachers-2026/similar.sh)
shows Qwen3.5-9B and Gemma 4 12B each text's 16 most similar of the 8 gold examples per class, then averages the
two as an ensemble, read from the score cache the single runs filled. On Banking77:

<!-- results: teachers-2026/similar/banking77.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77-qwen35-9b-similar16 | 16 similar examples, 4-bit `Qwen/Qwen3.5-9B` | 0.861 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.867 ± 0.004 | 0.867 ± 0.004 | 0.905 ± 0.001 | ONNX int8 | 0.9 | 33.0 | 23 |
| banking77-gemma4-12b-similar16 | 16 similar examples, 4-bit `google/gemma-4-12B-it` | 0.870 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.870 ± 0.005 | 0.869 ± 0.004 | 0.913 ± 0.001 | ONNX int8 | 1.2 | 37.5 | 23 |
| banking77-ensemble-similar16 | teacher ensemble of 2, 16 similar examples, 4-bit `Qwen/Qwen3.5-9B,google/gemma-4-12B-it` | 0.879 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.875 ± 0.003 | 0.875 ± 0.003 | 0.924 ± 0.001 | ONNX int8 | 1.4 | 40.6 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

The ensemble is the most accurate teacher on Banking77 so far (0.879, against 0.861 and 0.870 alone), and its
student reaches 0.875, level with the student of the 7B teacher with the probe; deployed in int8 it is 0.870
against that student's 0.872. It costs both teachers' labelling: 3.6 s per text on the GPU against 155 ms, so the
probe stays the Hub model.

On CLINC150, with the shortlist of 20 plus out of scope:

<!-- results: teachers-2026/similar/clinc150.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clinc150-qwen35-9b-similar16 | 16 similar examples, 4-bit `Qwen/Qwen3.5-9B` | 0.841 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.805 ± 0.005 | 0.848 ± 0.004 | 0.822 ± 0.003 | ONNX int8 | 1.0 | 23.7 | 23 |
| clinc150-gemma4-12b-similar16 | 16 similar examples, 4-bit `google/gemma-4-12B-it` | 0.914 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.850 ± 0.002 | 0.880 ± 0.002 | 0.863 ± 0.003 | ONNX int8 | 1.3 | 24.5 | 23 |
| clinc150-ensemble-similar16 | teacher ensemble of 2, 16 similar examples, 4-bit `Qwen/Qwen3.5-9B,google/gemma-4-12B-it` | 0.914 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.836 ± 0.001 | 0.873 ± 0.001 | 0.856 ± 0.001 | ONNX int8 | 1.1 | 24.7 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

Gemma 4 12B is far ahead of Qwen3.5-9B here (0.914 against 0.841), and averaging the two adds nothing to the
teacher (0.914) and costs the student: 0.836 against 0.850 for the student of Gemma alone, which reaches 0.849
deployed in int8, against 0.813 for the probe student of the 7B teacher. It took 8 hours of labelling on the
GPU, against 26 minutes for that probe run.

The probe on Gemma 4 12B
([`examples/teachers-2026/probe-gemma.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/teachers-2026/probe-gemma.sh)),
against the same teacher without it:

<!-- results: teachers-2026/probe/clinc150-gemma4-12b.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clinc150-gemma4-12b-similar16 | 16 similar examples, 4-bit `google/gemma-4-12B-it` | 0.914 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.850 ± 0.002 | 0.880 ± 0.002 | 0.863 ± 0.003 | ONNX int8 | 1.3 | 24.5 | 23 |
| clinc150-gemma4-12b-similar16-probe | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.941 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.860 ± 0.005 | 0.898 ± 0.004 | 0.870 ± 0.004 | ONNX int8 | 0.8 | 22.4 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

It lifts the teacher from 0.914 to 0.941 and the student from 0.850 to 0.860, 0.859 deployed in int8 (from
0.849), for about the same labelling time (7.2 hours). It is the new
[CLINC150 model on the Hub](https://huggingface.co/Shroomy/tinydistill-clinc150-minilm) and in the
[live demo](https://andreas-hum.github.io/tinydistill/demo/).

The same probe on Gemma 4 12B on Danish MASSIVE, with the English `all-MiniLM-L6-v2` student:

<!-- results: teachers-2026/probe/massive-da-gemma4-12b.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| massive-da-gemma4-12b-similar16 | 16 similar examples, 4-bit `google/gemma-4-12B-it` | 0.798 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.760 ± 0.003 | 0.706 ± 0.009 | 0.796 ± 0.005 | ONNX int8 | 0.9 | 28.8 | 23 |
| massive-da-gemma4-12b-similar16-probe | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.808 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.768 ± 0.004 | 0.721 ± 0.003 | 0.797 ± 0.003 | ONNX int8 | 1.1 | 31.9 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

The probe lifts the teacher from 0.798 to 0.808, the most accurate Danish teacher so far, and the English student
from 0.760 to 0.768 (0.763 deployed). The multilingual student does better on the same labels: 0.815 (below).

Without the probe, both teachers and their ensemble on Danish MASSIVE, with the English `all-MiniLM-L6-v2`
student:

<!-- results: teachers-2026/similar/massive-da.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| massive-da-qwen35-9b-similar16 | 16 similar examples, 4-bit `Qwen/Qwen3.5-9B` | 0.781 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.756 ± 0.002 | 0.700 ± 0.004 | 0.782 ± 0.002 | ONNX int8 | 0.9 | 30.4 | 23 |
| massive-da-gemma4-12b-similar16 | 16 similar examples, 4-bit `google/gemma-4-12B-it` | 0.798 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.760 ± 0.003 | 0.706 ± 0.009 | 0.796 ± 0.005 | ONNX int8 | 0.9 | 28.8 | 23 |
| massive-da-ensemble-similar16 | teacher ensemble of 2, 16 similar examples, 4-bit `Qwen/Qwen3.5-9B,google/gemma-4-12B-it` | 0.806 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.771 ± 0.003 | 0.710 ± 0.012 | 0.807 ± 0.003 | ONNX int8 | 0.8 | 29.1 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

Both teachers are well above the 8-shot Qwen2.5-7B-Instruct of the Danish section (0.781 and 0.798 against
0.736), and here the ensemble helps: 0.806 for the teacher, and 0.771 for its student, 0.764 deployed in int8,
against 0.724 for the same English student of the 8-shot 7B teacher. That is above the trimmed multilingual
student of the 7B teacher (0.761, 0.759 deployed), at the same size and over twice as fast, but it took 12 hours
of labelling with two teachers against 31 minutes with one.

The trimmed multilingual student on the ensemble's labels
([`examples/massive-da/ensemble-multilingual.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/massive-da/ensemble-multilingual.sh),
read from the score cache, so no new labelling):

<!-- results: massive-da/ensemble-multilingual.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| massive-da-ensemble-similar16 | teacher ensemble of 2, 16 similar examples, 4-bit `Qwen/Qwen3.5-9B,google/gemma-4-12B-it` | 0.806 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.771 ± 0.003 | 0.710 ± 0.012 | 0.807 ± 0.003 | ONNX int8 | 0.8 | 29.1 | 23 |
| paraphrase-multilingual-MiniLM-L12-v2-trimmed | 8-shot, 4-bit `Qwen/Qwen2.5-7B-Instruct` | 0.736 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 24 | 0.761 ± 0.001 | 0.708 ± 0.003 | 0.802 ± 0.006 | ONNX int8 | 1.8 | 40.2 | 24 |
| paraphrase-multilingual-MiniLM-L12-v2-trimmed-ensemble-similar16 | teacher ensemble of 2, 16 similar examples, 4-bit `Qwen/Qwen3.5-9B,google/gemma-4-12B-it` | 0.806 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 24 | 0.811 ± 0.002 | 0.783 ± 0.002 | 0.867 ± 0.003 | ONNX int8 | 1.9 | 46.2 | 24 |
| paraphrase-multilingual-MiniLM-L12-v2-trimmed-gemma4-12b-similar16-probe | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.808 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 24 | 0.815 ± 0.002 | 0.782 ± 0.005 | 0.849 ± 0.002 | ONNX int8 | 2.0 | 41.9 | 24 |

All latencies on the CPU with 2 threads.
<!-- /results -->

Here the better teachers pay off for the multilingual student: on the ensemble's labels 0.811, 0.810 deployed in
int8, against 0.761 on the 8-shot 7B teacher's labels, and above its own teacher (0.806). On the labels of Gemma
4 12B with the probe (above), 0.815 and 0.815 deployed, again above its teacher (0.808), with one teacher and
8.9 hours of labelling against two and 12.2 hours. That one is the Danish model on the Hub and in the live demo.

On financial tweets (3 classes, so the 16 similar examples are 16 of the 24 gold texts), with 5 epochs as in the
financial-tweets section:

<!-- results: teachers-2026/similar/financial-tweets.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| financial-tweets-qwen35-9b-similar16 | 16 similar examples, 4-bit `Qwen/Qwen3.5-9B` | 0.821 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.805 ± 0.005 | 0.752 ± 0.006 | 0.836 ± 0.004 | ONNX int8 | 1.2 | 66.9 | 23 |
| financial-tweets-gemma4-12b-similar16 | 16 similar examples, 4-bit `google/gemma-4-12B-it` | 0.846 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.803 ± 0.006 | 0.766 ± 0.008 | 0.837 ± 0.003 | ONNX int8 | 1.1 | 67.5 | 23 |
| financial-tweets-ensemble-similar16 | teacher ensemble of 2, 16 similar examples, 4-bit `Qwen/Qwen3.5-9B,google/gemma-4-12B-it` | 0.859 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.827 ± 0.008 | 0.788 ± 0.008 | 0.853 ± 0.001 | ONNX int8 | 1.3 | 67.6 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

The teachers rank as on the other tasks, the ensemble first (0.859), but Gemma 4 12B's student is no better than
Qwen3.5-9B's (0.803 against 0.805) for a teacher 2.5 points more accurate. The ensemble's student, 0.827 and 0.829
deployed in int8, ties the student of the zero-shot three-family ensemble of the financial-tweets section (0.828,
0.829 deployed), which needs no gold labels and took 6 minutes of labelling against an hour and a half.

## The probe alone: minutes of labelling instead of hours

With `label --probe-only` the probe is the whole teacher, and the teacher scores no label: it only runs the
prompt and each text to read the hidden state the probe needs.
[`examples/teachers-2026/probe-only.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/teachers-2026/probe-only.sh)
runs it with each task's best probe teacher, fit on the same 8 gold examples per class, once with a fixed
zero-shot prompt (encoded once for every text) and once with each text's 16 most similar gold examples. On
Banking77, with `Qwen/Qwen2.5-7B-Instruct`:

<!-- results: teachers-2026/probe-only/banking77.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77-7b-probe-only-zeroshot | zero-shot, 4-bit, probe only `Qwen/Qwen2.5-7B-Instruct` | 0.795 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.824 ± 0.002 | 0.822 ± 0.002 | 0.857 ± 0.004 | ONNX int8 | 1.2 | 38.4 | 23 |
| banking77-7b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `Qwen/Qwen2.5-7B-Instruct` | 0.854 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.870 ± 0.004 | 0.870 ± 0.004 | 0.892 ± 0.001 | ONNX int8 | 1.7 | 42.9 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

On CLINC150, with `google/gemma-4-12B-it`:

<!-- results: teachers-2026/probe-only/clinc150.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clinc150-gemma4-12b-probe-only-zeroshot | zero-shot, 4-bit, probe only `google/gemma-4-12B-it` | 0.891 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.842 ± 0.004 | 0.889 ± 0.002 | 0.842 ± 0.005 | ONNX int8 | 1.0 | 26.8 | 23 |
| clinc150-gemma4-12b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `google/gemma-4-12B-it` | 0.925 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.852 ± 0.002 | 0.896 ± 0.004 | 0.859 ± 0.002 | ONNX int8 | 1.6 | 34.1 | 23 |

All latencies on the CPU with 2 threads.
<!-- /results -->

On Danish MASSIVE, with `google/gemma-4-12B-it` and the trimmed multilingual student:

<!-- results: teachers-2026/probe-only/massive-da.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| massive-da-gemma4-12b-probe-only-zeroshot | zero-shot, 4-bit, probe only `google/gemma-4-12B-it` | 0.760 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 24 | 0.779 ± 0.002 | 0.744 ± 0.003 | 0.820 ± 0.003 | ONNX int8 | 2.4 | 45.3 | 24 |
| massive-da-gemma4-12b-probe-only-similar16 | 16 similar examples, 4-bit, probe only `google/gemma-4-12B-it` | 0.792 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 24 | 0.804 ± 0.002 | 0.777 ± 0.004 | 0.826 ± 0.003 | ONNX int8 | 1.6 | 40.6 | 24 |

All latencies on the CPU with 2 threads.
<!-- /results -->

Against the best run with the same teacher and examples, which also scores every label in context and
averages it with the probe (the probe sections above), the deployed students and the labelling times are:

- Banking77: 0.869 after 11.5 minutes with the similar examples and 0.824 after 74 seconds with the zero-shot
  prompt, against 0.872 after 35 minutes;
- CLINC150: 0.851 after 37 minutes and 0.837 after 10 minutes, against 0.859 after 7.2 hours;
- Danish MASSIVE: 0.802 after 27 minutes and 0.778 after 8 minutes, against 0.815 after 8.9 hours.

The probe alone with the 16 similar examples costs 0.3 to 1.3 points of the deployed student and a third to a
twentieth of the labelling time; scoring the labels is what takes long, most of all for teachers that cannot
pack texts (Gemma 4) and with many labels. With the fixed prompt, labelling takes minutes and the student loses
1.4 to 4.5 points more. Both need gold labels to fit on (here 8 per class) and a local teacher. The times are
the GPU time `label` measured (RTX 4090, 4 bit); for Danish MASSIVE, the labelling of the Gemma 4 12B probe run
whose cache the multilingual student read (the same in context and probe labels).

## Which layer the probe reads

By default the probe reads the input of the teacher's LM head, the last layer's output; `label --probe-layer L`
reads the output of layer L instead.
[`examples/teachers-2026/probe-layers.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/teachers-2026/probe-layers.sh)
measures the probe alone (`--probe-only`, zero-shot prompt, 8 gold examples per class) for the default and five
layers nearer the middle. `Qwen/Qwen2.5-7B-Instruct` has 28 layers and `google/gemma-4-12B-it` 48. On Banking77:

<!-- results: teachers-2026/probe-layers/banking77.md -->
| Teacher | Prompt | Probe reads | Teacher accuracy | Macro-F1 | Labelling |
|---|---|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | zero-shot | LM head input (default) | 0.795 | 0.792 | from the cache |
| `Qwen/Qwen2.5-7B-Instruct` | zero-shot | layer -4 | 0.806 | 0.803 | 60 s |
| `Qwen/Qwen2.5-7B-Instruct` | zero-shot | layer -7 | 0.811 | 0.809 | 61 s |
| `Qwen/Qwen2.5-7B-Instruct` | zero-shot | layer -10 | 0.779 | 0.777 | 60 s |
| `Qwen/Qwen2.5-7B-Instruct` | zero-shot | layer -14 | 0.492 | 0.489 | 60 s |
| `Qwen/Qwen2.5-7B-Instruct` | zero-shot | layer -18 | 0.386 | 0.385 | 59 s |

Test accuracy of the probe alone (`label --probe-only`) on 3,080 test texts.
<!-- /results -->

On CLINC150:

<!-- results: teachers-2026/probe-layers/clinc150.md -->
| Teacher | Prompt | Probe reads | Teacher accuracy | Macro-F1 | Labelling |
|---|---|---|---|---|---|
| `google/gemma-4-12B-it` | zero-shot | LM head input (default) | 0.891 | 0.914 | 528 s |
| `google/gemma-4-12B-it` | zero-shot | layer -6 | 0.886 | 0.911 | 517 s |
| `google/gemma-4-12B-it` | zero-shot | layer -12 | 0.889 | 0.915 | 522 s |
| `google/gemma-4-12B-it` | zero-shot | layer -18 | 0.859 | 0.880 | 549 s |
| `google/gemma-4-12B-it` | zero-shot | layer -24 | 0.687 | 0.743 | 564 s |
| `google/gemma-4-12B-it` | zero-shot | layer -30 | 0.673 | 0.723 | 589 s |

Test accuracy of the probe alone (`label --probe-only`) on 5,500 test texts.
<!-- /results -->

On Danish MASSIVE:

<!-- results: teachers-2026/probe-layers/massive-da.md -->
| Teacher | Prompt | Probe reads | Teacher accuracy | Macro-F1 | Labelling |
|---|---|---|---|---|---|
| `google/gemma-4-12B-it` | zero-shot | LM head input (default) | 0.760 | 0.740 | from the cache |
| `google/gemma-4-12B-it` | zero-shot | layer -6 | 0.764 | 0.744 | 1116 s |
| `google/gemma-4-12B-it` | zero-shot | layer -12 | 0.775 | 0.752 | 874 s |
| `google/gemma-4-12B-it` | zero-shot | layer -18 | 0.722 | 0.691 | 838 s |
| `google/gemma-4-12B-it` | zero-shot | layer -24 | 0.538 | 0.493 | 841 s |
| `google/gemma-4-12B-it` | zero-shot | layer -30 | 0.537 | 0.500 | 918 s |

Test accuracy of the probe alone (`label --probe-only`) on 2,974 test texts.
<!-- /results -->

A layer about a quarter of the way back is a little better on two tasks (Banking77: 0.811 at layer -7 against
0.795; Danish MASSIVE: 0.775 at layer -12 against 0.760) and a little worse on the third (CLINC150: 0.889 at
layer -12 against 0.891). Nearer the middle the probe falls far behind on every task (0.386, 0.673 and 0.537 at
the furthest layer measured). The default stays the LM head's input; with `--probe-only` a few layers can be
compared in minutes per task.

## A larger student: when the teacher is far ahead

Every run above trains MiniLM-L6 (23M parameters). `train --out` trains another student on a run's labels
without running the teacher again. On CLINC150, where Gemma 4 12B with the probe is far above its student
(0.941 against 0.860), [`examples/clinc150/students.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/clinc150/students.sh)
trains MiniLM-L12 and mpnet-base on the same labels:

<!-- results: clinc150/students/summary.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clinc150-gemma4-12b-similar16-probe | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.941 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.860 ± 0.005 | 0.898 ± 0.004 | 0.870 ± 0.004 | ONNX int8 | 0.8 | 22.4 | 23 |
| all-MiniLM-L12-v2 | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.941 | `sentence-transformers/all-MiniLM-L12-v2` | 33 | 0.872 ± 0.006 | 0.907 ± 0.003 | 0.880 ± 0.006 | ONNX int8 | 2.1 | 42.6 | 34 |
| all-mpnet-base-v2 | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.941 | `sentence-transformers/all-mpnet-base-v2` | 110 | 0.894 ± 0.005 | 0.921 ± 0.003 | 0.905 ± 0.003 | ONNX int8 | 7.2 | 131.2 | 110 |

All latencies on the CPU with 2 threads.
<!-- /results -->

mpnet-base closes 0.034 of the 0.081 gap: 0.894 (0.902 deployed in int8, against 0.859), for 7 ms per text on a CPU
instead of 0.8. Most of that gain is on the out-of-scope requests: on the 4,500 in-scope texts, scored as
[above](#scored-as-the-few-shot-papers-score-it-150-intents), mpnet reaches 0.960 ± 0.001 (0.948 deployed)
against 0.948 ± 0.002 (0.945) for MiniLM-L6 and 0.954 ± 0.000 (0.952) for MiniLM-L12
([`in-scope.json`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/clinc150/in-scope.json)).
On the other tasks, whose students already match or beat their teachers,
[`examples/students/mpnet.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/students/mpnet.sh)
trains mpnet on each task's best labels:

<!-- results: students/mpnet.md -->
| Run | Teacher | Teacher accuracy | Student | Parameters (M) | Accuracy | Macro-F1 | Agreement with teacher | Deployed as | Latency, batch 1 (ms) | Latency, batch 32 (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| banking77-7b-similar16-probe | 16 similar examples, 4-bit, probe `Qwen/Qwen2.5-7B-Instruct` | 0.859 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.875 ± 0.001 | 0.875 ± 0.001 | 0.890 ± 0.001 | ONNX int8 | 1.0 | 33.7 | 23 |
| banking77-mpnet | 16 similar examples, 4-bit, probe `Qwen/Qwen2.5-7B-Instruct` | 0.859 | `sentence-transformers/all-mpnet-base-v2` | 110 | 0.882 ± 0.001 | 0.882 ± 0.001 | 0.898 ± 0.003 | ONNX int8 | 7.4 | 198.6 | 110 |
| hwu64-gemma4-12b-similar16 | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.877 | `sentence-transformers/all-MiniLM-L6-v2` | 23 | 0.894 ± 0.005 | 0.893 ± 0.004 | 0.893 ± 0.003 | ONNX int8 | 0.8 | 21.7 | 23 |
| hwu64-mpnet | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.877 | `sentence-transformers/all-mpnet-base-v2` | 110 | 0.891 ± 0.002 | 0.891 ± 0.001 | 0.904 ± 0.002 | ONNX int8 | 5.3 | 113.8 | 110 |
| paraphrase-multilingual-MiniLM-L12-v2-trimmed-gemma4-12b-similar16-probe | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.808 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 24 | 0.815 ± 0.002 | 0.782 ± 0.005 | 0.849 ± 0.002 | ONNX int8 | 2.0 | 41.9 | 24 |
| massive-da-multilingual-mpnet | 16 similar examples, 4-bit, probe `google/gemma-4-12B-it` | 0.808 | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 90 | 0.818 ± 0.004 | 0.793 ± 0.004 | 0.861 ± 0.003 | ONNX int8 | 5.6 | 119.5 | 91 |

All latencies on the CPU with 2 threads.
<!-- /results -->

Here mpnet gains less than a point (Banking77 0.882 against 0.875, Danish 0.818 against 0.815) or nothing
(HWU64 0.891 against 0.894), and its int8 export loses accuracy on two of them (Banking77 0.868 and HWU64 0.864,
against 0.882 and 0.891 for the fp32 export), so check the fp32 row before deploying it in int8. A larger
student pays when the teacher is well above the student, and otherwise the teacher is the limit.

## A vLLM teacher agrees with the local one

[`examples/vllm-smoke.sh`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/vllm-smoke.sh)
labels 300 financial tweets with `Qwen/Qwen2.5-1.5B-Instruct` locally and through a vLLM server:

<!-- results: vllm-smoke.md -->
| Texts | Same top label | Largest difference in a label score | Mean difference |
|---|---|---|---|
| 300 | 100.0% | 0.611 | 0.0900 |

Qwen2.5-1.5B-Instruct scoring 300 financial tweets locally (bf16, tinydistill) and through vLLM (prompt log-probabilities); label scores are log-probabilities, so small differences are numerical.
<!-- /results -->
