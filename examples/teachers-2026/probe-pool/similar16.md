| Run | Teacher | Layer | last token, LM head input (default) | mean, LM head input | last token, layer | mean, layer |
|---|---|---|---|---|---|---|
| banking77-gemma4-12b-probe-only-similar16 | `google/gemma-4-12B-it` | -12 of 48 | 0.871 | 0.864 | 0.870 | 0.877 |
| clinc150-gemma4-12b-probe-only-similar16 | `google/gemma-4-12B-it` | -12 of 48 | 0.924 | 0.894 | 0.927 | 0.923 |
| hwu64-gemma4-12b-probe-only-similar16 | `google/gemma-4-12B-it` | -12 of 48 | 0.875 | 0.864 | 0.876 | 0.893 |
| massive-da-gemma4-12b-probe-only-similar16 | `google/gemma-4-12B-it` | -12 of 48 | 0.787 | 0.740 | 0.801 | 0.808 |
| trec50-gemma4-12b-probe-only-similar16 | `google/gemma-4-12B-it` | -12 of 48 | 0.728 | 0.756 | 0.770 | 0.820 |

Test accuracy of the probe alone, fit on the same gold texts, reading the teacher's state at the text's last token or the mean of its states over the text's own tokens (the text and the prompt after it, not the examples before it), at the LM head's input or at the output of the layer given.
