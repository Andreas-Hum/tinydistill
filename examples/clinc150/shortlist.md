| Teacher scores | Labelling time | Teacher accuracy | In scope | Out of scope | Gold label scored | Teacher latency (GPU) | Student accuracy (int8, CPU) |
|---|---|---|---|---|---|---|---|
| every label (151) | 95 min | 0.564 | 0.666 | 0.107 | 100% | 262 ms | 0.559 |
| `--shortlist 20` | 15 min | 0.555 | 0.674 | 0.020 | 80.3% | 73 ms | 0.555 |
| `--shortlist 20 --shortlist-keep oos` | 15 min | 0.581 | 0.672 | 0.175 | 96.8% | 87 ms | 0.574 |

CLINC150 (plus), 5,500 test texts of which 1,000 are out of scope; `Qwen/Qwen2.5-7B-Instruct` in 4 bit, zero-shot, labelling about 20,748 training and test texts on one GPU. Gold label scored: the share of test texts whose gold label the teacher scored at all.
