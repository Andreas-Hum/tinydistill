| Texts the teacher labels | Without warm-up | With `--warmup-pairs 20000` |
|---|---|---|
| 1,000 of 10,003, at random | 0.480 ± 0.017 | 0.510 ± 0.023 |
| 2,500 of 10,003, at random | 0.735 ± 0.011 | 0.739 ± 0.010 |
| all 10,003 | 0.857 ± 0.001 | 0.855 ± 0.002 |
| all 10,003, deployed (ONNX int8) | 0.855 | 0.850 |

Banking77, 3080 test texts; the teacher of similar.sh (16 most similar examples), its labels read from the score cache. Student accuracy (PyTorch unless marked), mean ± standard deviation over 3 training seeds.
