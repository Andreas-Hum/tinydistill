| Run | Teacher | Layer | last token, LM head input (default) | mean, LM head input | last token, layer | mean, layer |
|---|---|---|---|---|---|---|
| banking77-7b-probe-only-zeroshot | `Qwen/Qwen2.5-7B-Instruct` | -7 of 28 | 0.797 | 0.768 | 0.810 | 0.760 |
| clinc150-gemma4-12b-probe-only-zeroshot | `google/gemma-4-12B-it` | -12 of 48 | 0.892 | 0.858 | 0.889 | 0.908 |
| hwu64-gemma4-12b-probe-only-zeroshot | `google/gemma-4-12B-it` | -12 of 48 | 0.848 | 0.833 | 0.857 | 0.889 |
| massive-da-gemma4-12b-probe-only-zeroshot | `google/gemma-4-12B-it` | -12 of 48 | 0.763 | 0.711 | 0.774 | 0.802 |
| trec50-gemma4-12b-probe-only-zeroshot | `google/gemma-4-12B-it` | -12 of 48 | 0.722 | 0.754 | 0.742 | 0.766 |

Test accuracy of the probe alone, fit on the same gold texts, reading the teacher's state at the text's last token or the mean of its states over the text's own tokens (the text and the prompt after it, not the examples before it), at the LM head's input or at the output of the layer given.
