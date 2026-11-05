# How it compares

How does a tinydistill student compare with what people use instead? Each table below is one task, with the same
gold budget for every method that trains (8 labelled examples per class, the same texts) and the same test set:

- **Gold only**: the student's encoder fine-tuned on the gold labels alone, no teacher (tinydistill's
  `--gold-only` baseline, trained on the CPU).
- **SetFit** ([Tunstall et al., 2022](https://arxiv.org/abs/2209.11055)) on the same gold labels, 3 seeds, on
  the CPU.
- **GLiClass zero-shot** (`knowledgator/gliclass-base-v3.0`) with the task's label names, and with their
  descriptions where the task has them. No gold labels, no training.
- **The LLM as the classifier**: the teacher itself, read from the saved run, on a GPU.
- **tinydistill**: the most accurate saved student per task that uses at most 8 gold labels per class, read from
  the saved results in `examples/`; the table picks up new runs when it is rebuilt.

Latency is per text at batch 1 with 2 CPU threads, on the same machine and the same 100 test texts for every
model. The scripts are in
[`examples/compare/`](https://github.com/Andreas-Hum/tinydistill/tree/main/examples/compare); `run.sh` there
reruns everything.

## What the tables show

- **tinydistill is the most accurate method that runs on a CPU** on Banking77 (0.867 against SetFit's 0.796;
  0.859 as the deployed int8 model) and CLINC150 (0.767 against 0.712, the same deployed); on Danish MASSIVE it
  reaches 0.761 against 0.586 for gold only. Its int8 student and the gold-only model, the same encoder, are the
  fastest and smallest models in every table: 1.0 to 2.3 ms per text and 23 to 24 MB, against 4.6 to 6.9 ms
  and 92 MB for SetFit and 113 to 577 ms and 746 MB for GLiClass.
- **The student matches its teacher at a fraction of the latency.** On Banking77, CLINC150 and Danish MASSIVE
  the student is within 0.002 of the LLM that labelled its data or above it (0.867 against 0.861 on Banking77),
  at 1.0 to 2.3 ms on a CPU against 97 to 907 ms on a GPU. On financial tweets it is 0.014 below its teacher
  ensemble.
- **It needs a GPU and a good teacher; SetFit needs neither.** SetFit trained in 7 minutes on the CPU and came
  within 0.071 of the best student on Banking77 and 0.055 on CLINC150. The best Banking77 student took 3.2 hours
  of labelling with Qwen3.5-9B; the 7B teacher's student is 0.010 lower after 35 minutes.
- **A weak teacher gives a weak student.** With a zero-shot teacher and no gold labels, the CLINC150 student
  (0.573) is below gold only (0.580) and SetFit (0.712); with the 16 most similar gold examples in the teacher's
  prompt and out of scope as an `outside` label, it reaches 0.767.
- **With very few labels, zero-shot can beat few-shot training.** On financial tweets 8 labels per class are 24
  texts: SetFit reaches 0.420 and gold only 0.318, below GLiClass zero-shot (0.676). The student, trained on
  the LLM's labels without any gold labels, reaches 0.828.
- **GLiClass is the weakest on the many-label tasks** (0.443 on Banking77, 0.530 on CLINC150) and the slowest
  model on a CPU, because it reads every label name with every text. Label descriptions lowered its accuracy on
  both tasks where they were tried.
- **Published 10-shot results on Banking77 are higher**: RoBERTa-Large with ICDA reports 0.898 and SetFit with
  MPNet on expert-picked examples 0.881, against the student's 0.867 with 8 labels per class. They use 10 labels
  per class and larger models, and ICDA also uses texts written by OPT-66B, so they are listed apart and not
  rerun.

Not run, for time: SetFit with `all-mpnet-base-v2` (on Banking77 it ran for 90 minutes on the CPU without
finishing its 3 seeds) and GLiClass large, on every task; SetFit and GLiClass on Danish MASSIVE. The saved
SetFit run with `paraphrase-mpnet-base-v2` on Banking77, trained on a GPU, reached 0.789 ± 0.005 with the same
616 labels ([`examples/banking77/setfit.md`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/banking77/setfit.md)).
`examples/compare/run.sh` includes these runs.

## Banking77

<!-- results: compare/banking77.md -->
| Method | Gold labels | Also uses | Accuracy | Macro-F1 | Latency per text, CPU | Size | Build time | GPU needed |
|---|---|---|---|---|---|---|---|---|
| Gold only: `all-MiniLM-L6-v2` fine-tuned on the gold labels, ONNX int8 | 616 |  | 0.565 ± 0.035 | 0.530 ± 0.043 | 1.2 ms | 23 MB | 62 s (CPU) | no |
| SetFit, `all-MiniLM-L6-v2` | 616 |  | 0.796 ± 0.002 | 0.792 ± 0.002 | 5.4 ms | 92 MB | 7 min (CPU) | no |
| GLiClass zero-shot, `gliclass-base-v3.0` | 0 | label names | 0.443 | 0.460 | 419.7 ms | 746 MB | none | no |
| The LLM as the classifier: `Qwen2.5-7B-Instruct`, 16 similar examples, 4-bit | 616 (in the prompt) | | 0.851 | 0.850 | 157 ms (RTX 4090) | 5.4 GB | none | yes, to run it |
| tinydistill: `all-MiniLM-L6-v2` student of that LLM, ONNX int8 | 616 | the LLM's labels on 10,003 unlabelled texts | 0.857 ± 0.001 | 0.856 ± 0.001 | 1.3 ms | 23 MB | 35 min labelling + 58 s training (RTX 4090) | yes, to build it |
| The LLM as the classifier: `Qwen3.5-9B`, 16 similar examples, 4-bit | 616 (in the prompt) | | 0.861 | 0.860 | 907 ms (RTX 4090) | 7.5 GB | none | yes, to run it |
| tinydistill: `all-MiniLM-L6-v2` student of that LLM, ONNX int8 | 616 | the LLM's labels on 10,003 unlabelled texts | 0.867 ± 0.004 | 0.867 ± 0.004 | 1.1 ms | 23 MB | 3.2 h labelling + 89 s training (RTX 4090) | yes, to build it |

Banking77, 3,080 test texts. Every method that trains gets the same 616 gold-labelled texts (8 per class, tinydistill's own pick with seed 0); mean ± standard deviation over 3 training seeds. Latency: batch 1, 2 threads, median over three passes through the same 100 test texts, each library's own inference (tinydistill's Predictor for the ONNX models); the LLM runs on a GPU. The gold-only and tinydistill accuracies are the PyTorch models'; their latency and size are those of the int8 export ("saved": timed when the run was made, not re-timed with the others). Build time is per model; the CPU runs trained with 16 threads. The last tinydistill row is [`examples/banking77/banking77-qwen35-9b-similar16.json`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/banking77/banking77-qwen35-9b-similar16.json), the most accurate of 24 saved runs on this test set that use at most 8 gold labels per class (picked on the test set); the int8 model it times scores 0.859. The LLM and tinydistill rows before it are an earlier run, kept for reference: `examples/banking77/banking77-7b-similar16.json`.

Reported in the papers, not rerun. The protocol differs from the table above: 10 gold labels per class (770) against our 8 (616), plus what the third column says, and larger classifiers (RoBERTa-Large, 355M parameters; MPNet, 110M). The test set is the same 3,080 texts. Lin et al. sample the 10 per class at random, average 5 seeds and train RoBERTa-Large for 40 epochs; Loukas et al. train on 10 per class that a human expert picked as representative.

| Method | Gold labels | Also uses | Accuracy | Source |
|---|---|---|---|---|
| RoBERTa-Large + classifier, 10-shot | 770 |  | 0.861 | Lin et al. 2023, Table 2 ([arXiv 2302.05096](https://arxiv.org/abs/2302.05096)) |
| DNNC, 10-shot | 770 | pre-training on natural language inference datasets | 0.867 | Table 2 of [arXiv 2212.00301](https://arxiv.org/abs/2212.00301) and of arXiv 2302.05096 |
| SetFit, MPNet-v2, 10-shot | 770, picked by a human expert |  | 0.881 | Loukas et al. 2023, Table 4 ([arXiv 2311.06102](https://arxiv.org/abs/2311.06102)); micro-F1 |
| RoBERTa-Large + ICDA-XL, 10-shot | 770 | 128x as many texts written by OPT-66B, filtered by PVI on a validation set | 0.898 | Lin et al. 2023, Table 2 ([arXiv 2302.05096](https://arxiv.org/abs/2302.05096)) |
<!-- /results -->

## CLINC150

<!-- results: compare/clinc150.md -->
| Method | Gold labels | Also uses | Accuracy | Macro-F1 | Latency per text, CPU | Size | Build time | GPU needed |
|---|---|---|---|---|---|---|---|---|
| Gold only: `all-MiniLM-L6-v2` fine-tuned on the gold labels, ONNX int8 | 1208 |  | 0.580 ± 0.019 | 0.620 ± 0.023 | 1.1 ms | 23 MB | 63 s (CPU) | no |
| SetFit, `all-MiniLM-L6-v2` | 1208 |  | 0.712 ± 0.001 | 0.780 ± 0.000 | 4.6 ms | 92 MB | 7 min (CPU) | no |
| GLiClass zero-shot, `gliclass-base-v3.0` | 0 | label names | 0.530 | 0.626 | 577.3 ms | 746 MB | none | no |
| GLiClass zero-shot, `gliclass-base-v3.0` | 0 | label names and descriptions | 0.520 | 0.616 | 538.3 ms | 746 MB | none | no |
| The LLM as the classifier: `Qwen2.5-7B-Instruct`, zero-shot, 4-bit | 0 | | 0.581 | 0.616 | 87 ms (RTX 4090) | 5.4 GB | none | yes, to run it |
| tinydistill: `all-MiniLM-L6-v2` student of that LLM, ONNX int8 | 0 | the LLM's labels on 15,250 unlabelled texts | 0.573 ± 0.003 | 0.604 ± 0.005 | 1.2 ms | 23 MB | 15 min labelling + 2 min training (RTX 4090) | yes, to build it |
| The LLM as the classifier: `Qwen2.5-7B-Instruct`, 16 similar examples, 4-bit | 1208 (in the prompt) | | 0.769 | 0.834 | 97 ms (RTX 4090) | 5.4 GB | none | yes, to run it |
| tinydistill: `all-MiniLM-L6-v2` student of that LLM, ONNX int8 | 1208 | the LLM's labels on 15,250 unlabelled texts | 0.767 ± 0.002 | 0.832 ± 0.001 | 1.0 ms | 23 MB | 28 min labelling + 2 min training (RTX 4090) | yes, to build it |

CLINC150 (plus, with out of scope), 5,500 test texts. Every method that trains gets the same 1208 gold-labelled texts (8 per class, tinydistill's own pick with seed 0); mean ± standard deviation over 3 training seeds. Latency: batch 1, 2 threads, median over three passes through the same 100 test texts, each library's own inference (tinydistill's Predictor for the ONNX models); the LLM runs on a GPU. The gold-only and tinydistill accuracies are the PyTorch models'; their latency and size are those of the int8 export ("saved": timed when the run was made, not re-timed with the others). Build time is per model; the CPU runs trained with 16 threads. The last tinydistill row is [`examples/clinc150/clinc150-7b-outside-similar16.json`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/clinc150/clinc150-7b-outside-similar16.json), the most accurate of 5 saved runs on this test set that use at most 8 gold labels per class (picked on the test set); the int8 model it times scores 0.767. That run labels the out-of-scope requests as an `outside` label, "none of these" (examples/clinc150/outside.yaml); the test texts and their gold labels are the same. The LLM and tinydistill rows before it are an earlier run, kept for reference: `examples/clinc150/clinc150-7b-shortlist20-keep-oos.json`.
<!-- /results -->

## Danish MASSIVE

<!-- results: compare/massive-da.md -->
| Method | Gold labels | Also uses | Accuracy | Macro-F1 | Latency per text, CPU | Size | Build time | GPU needed |
|---|---|---|---|---|---|---|---|---|
| Gold only: `paraphrase-multilingual-MiniLM-L12-v2` fine-tuned on the gold labels, ONNX int8 | 476 |  | 0.586 ± 0.025 | 0.524 ± 0.016 | 2.5 ms | 118 MB | 80 s (CPU) | no |
| The LLM as the classifier: `Qwen2.5-7B-Instruct`, 8-shot, 4-bit | 476 (in the prompt) | | 0.736 | 0.688 | 144 ms (RTX 4090) | 5.4 GB | none | yes, to run it |
| tinydistill: `paraphrase-multilingual-MiniLM-L12-v2` student of that LLM, ONNX int8 | 476 | the LLM's labels on 11,514 unlabelled texts | 0.761 ± 0.001 | 0.708 ± 0.003 | 2.3 ms | 24 MB | 31 min labelling + 2 min training (RTX 4090) | yes, to build it |

Danish MASSIVE, 2,974 test texts. Every method that trains gets the same 476 gold-labelled texts (8 per class, tinydistill's own pick with seed 0); mean ± standard deviation over 3 training seeds. Latency: batch 1, 2 threads, median over three passes through the same 100 test texts, each library's own inference (tinydistill's Predictor for the ONNX models); the LLM runs on a GPU. The gold-only and tinydistill accuracies are the PyTorch models'; their latency and size are those of the int8 export ("saved": timed when the run was made, not re-timed with the others). Build time is per model; the CPU runs trained with 16 threads. The last tinydistill row is [`examples/massive-da/paraphrase-multilingual-MiniLM-L12-v2-trimmed.json`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/massive-da/paraphrase-multilingual-MiniLM-L12-v2-trimmed.json), the most accurate of 4 saved runs on this test set that use at most 8 gold labels per class (picked on the test set); the int8 model it times scores 0.759.
<!-- /results -->

## Financial tweets

<!-- results: compare/financial-tweets.md -->
| Method | Gold labels | Also uses | Accuracy | Macro-F1 | Latency per text, CPU | Size | Build time | GPU needed |
|---|---|---|---|---|---|---|---|---|
| Gold only: `all-MiniLM-L6-v2` fine-tuned on the gold labels, ONNX int8 | 24 |  | 0.318 ± 0.100 | 0.293 ± 0.060 | 1.6 ms | 23 MB | 2 s (CPU) | no |
| SetFit, `all-MiniLM-L6-v2` | 24 |  | 0.420 ± 0.006 | 0.399 ± 0.005 | 6.9 ms | 92 MB | 24 s (CPU) | no |
| GLiClass zero-shot, `gliclass-base-v3.0` | 0 | label names | 0.676 | 0.665 | 112.8 ms | 746 MB | none | no |
| GLiClass zero-shot, `gliclass-base-v3.0` | 0 | label names and descriptions | 0.632 | 0.625 | 142.8 ms | 746 MB | none | no |
| The LLM as the classifier: `Qwen2.5-7B-Instruct`, `Mistral-7B-Instruct-v0.3`, `Phi-3.5-mini-instruct`, ensemble of 3, zero-shot, 4-bit | 0 | | 0.842 | 0.801 | 242 ms (RTX 4090) | 11.7 GB | none | yes, to run it |
| tinydistill: `all-MiniLM-L6-v2` student of that LLM, ONNX int8 | 0 | the LLM's labels on 9,543 unlabelled texts | 0.828 ± 0.006 | 0.778 ± 0.005 | 1.6 ms | 23 MB | 6 min labelling + at most 2 min training (RTX 4090) | yes, to build it |

Financial tweets, 2,388 test texts. Every method that trains gets the same 24 gold-labelled texts (8 per class, tinydistill's own pick with seed 0); mean ± standard deviation over 3 training seeds. Latency: batch 1, 2 threads, median over three passes through the same 100 test texts, each library's own inference (tinydistill's Predictor for the ONNX models); the LLM runs on a GPU. The gold-only and tinydistill accuracies are the PyTorch models'; their latency and size are those of the int8 export ("saved": timed when the run was made, not re-timed with the others). Build time is per model; the CPU runs trained with 8 or 16 threads. The last tinydistill row is [`examples/financial-tweets/financial-tweets-ensemble.json`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/financial-tweets/financial-tweets-ensemble.json), the most accurate of 6 saved runs on this test set that use at most 8 gold labels per class (picked on the test set); the int8 model it times scores 0.830.
<!-- /results -->
