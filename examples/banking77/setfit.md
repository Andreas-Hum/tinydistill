| Method | Gold labels | Also uses | Accuracy | Macro-F1 |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct, 8 examples per class in the prompt | 616 | | 0.783 | 0.779 |
| SetFit, MiniLM-L6 | 616 | | 0.797 ± 0.001 | 0.793 ± 0.002 |
| SetFit, mpnet-base | 616 | | 0.789 ± 0.005 | 0.788 ± 0.005 |
| tinydistill, MiniLM-L6 student | 616 | the 7B teacher's soft labels on 10,003 texts | 0.813 ± 0.002 | 0.811 ± 0.003 |
| tinydistill, static student | 616 | the same | 0.803 ± 0.001 | 0.801 ± 0.001 |
| MiniLM-L6 trained on all gold labels (upper bound) | 10,003 | | 0.914 ± 0.005 | 0.914 ± 0.005 |

Banking77, 3080 test texts; mean ± standard deviation over three seeds. Every method sees the same 616 gold-labelled texts (8 per class). SetFit: one epoch of contrastive training with 20 pair iterations, then a logistic-regression head (the library's defaults otherwise). The tinydistill students train for 20 epochs; the upper bound for 5.
