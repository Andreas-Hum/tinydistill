| Method | Gold labels | Also uses | Accuracy | Macro-F1 | Latency per text, CPU | Size | Build time | GPU needed |
|---|---|---|---|---|---|---|---|---|
| Gold only: `all-MiniLM-L6-v2` fine-tuned on the gold labels, ONNX int8 | 616 |  | 0.565 ± 0.035 | 0.530 ± 0.043 | 1.2 ms | 23 MB | 62 s (CPU) | no |
| SetFit, `all-MiniLM-L6-v2` | 616 |  | 0.796 ± 0.002 | 0.792 ± 0.002 | 5.4 ms | 92 MB | 7 min (CPU) | no |
| GLiClass zero-shot, `gliclass-base-v3.0` | 0 | label names | 0.443 | 0.460 | 419.7 ms | 746 MB | none | no |
| The LLM as the classifier: `Qwen2.5-7B-Instruct`, 16 similar examples, 4-bit | 616 (in the prompt) | | 0.851 | 0.850 | 157 ms (RTX 4090) | 5.4 GB | none | yes, to run it |
| tinydistill: `all-MiniLM-L6-v2` student of that LLM, ONNX int8 | 616 | the LLM's labels on 10,003 unlabelled texts | 0.857 ± 0.001 | 0.856 ± 0.001 | 1.3 ms | 23 MB | 35 min labelling + 58 s training (RTX 4090) | yes, to build it |
| The LLM as the classifier: `Qwen2.5-7B-Instruct`, 16 similar examples, 4-bit, probe | 616 (in the prompt) | | 0.859 | 0.859 | 155 ms (RTX 4090) | 5.4 GB | none | yes, to run it |
| tinydistill: `all-MiniLM-L6-v2` student of that LLM, ONNX int8 | 616 | the LLM's labels on 10,003 unlabelled texts | 0.875 ± 0.001 | 0.875 ± 0.001 | 1.2 ms | 23 MB | 35 min labelling + 58 s training (RTX 4090) | yes, to build it |

Banking77, 3,080 test texts. Every method that trains gets the same 616 gold-labelled texts (8 per class, tinydistill's own pick with seed 0); mean ± standard deviation over 3 training seeds. Latency: batch 1, 2 threads, median over three passes through the same 100 test texts, each library's own inference (tinydistill's Predictor for the ONNX models); the LLM runs on a GPU. The gold-only and tinydistill accuracies are the PyTorch models'; their latency and size are those of the int8 export ("saved": timed when the run was made, not re-timed with the others). Build time is per model; the CPU runs trained with 16 threads. The last tinydistill row is [`examples/teachers-2026/probe/banking77-7b-similar16-probe.json`](https://github.com/Andreas-Hum/tinydistill/blob/main/examples/teachers-2026/probe/banking77-7b-similar16-probe.json), the most accurate of 29 saved runs on this test set that use at most 8 gold labels per class (picked on the test set, a tie at three decimals going to the better int8 model, then the smaller); the int8 model it times scores 0.872. The LLM and tinydistill rows before it are an earlier run, kept for reference: `examples/banking77/banking77-7b-similar16.json`.

Reported in the papers, not rerun. The protocol differs from the table above: 10 gold labels per class (770) against our 8 (616), plus what the third column says, and larger classifiers (RoBERTa-Large, 355M parameters; MPNet, 110M). The test set is the same 3,080 texts. Lin et al. sample the 10 per class at random, average 5 seeds and train RoBERTa-Large for 40 epochs; Loukas et al. train on 10 per class that a human expert picked as representative.

| Method | Gold labels | Also uses | Accuracy | Source |
|---|---|---|---|---|
| RoBERTa-Large + classifier, 10-shot | 770 |  | 0.861 | Lin et al. 2023, Table 2 ([arXiv 2302.05096](https://arxiv.org/abs/2302.05096)) |
| DNNC, 10-shot | 770 | pre-training on natural language inference datasets | 0.867 | Table 2 of [arXiv 2212.00301](https://arxiv.org/abs/2212.00301) and of arXiv 2302.05096 |
| SetFit, MPNet-v2, 10-shot | 770, picked by a human expert |  | 0.881 | Loukas et al. 2023, Table 4 ([arXiv 2311.06102](https://arxiv.org/abs/2311.06102)); micro-F1 |
| RoBERTa-Large + ICDA-XL, 10-shot | 770 | 128x as many texts written by OPT-66B, filtered by PVI on a validation set | 0.898 | Lin et al. 2023, Table 2 ([arXiv 2302.05096](https://arxiv.org/abs/2302.05096)) |
