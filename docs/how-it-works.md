# How it works

## Scoring the labels

The teacher never generates free text. For an input x, the prompt is rendered with the model's chat template
(the user message is the instruction with the label list, followed by the generation prompt for the assistant).
A continuation is one way of writing a label followed by the end-of-turn token, c<sub>1</sub> … c<sub>m</sub>,
with log-probability

    log p(c | prompt(x)) = sum_j log p(c_j | prompt(x), c_1 ... c_(j-1))

Multi-token labels are handled exactly, and the end-of-turn token keeps a label that is a prefix of another
(`card` versus `card arrival`) from winning by stopping early.

Models do not always write a label the way it is listed: Qwen2.5 tends to answer a sentiment question with
"Bullish" even when the list says "bullish". So each label is scored over its common surface forms, the name as
written, lowercase, Capitalised and UPPERCASE, each with and without a leading space (forms that tokenise
identically count once):

    score(label) = log sum over forms c of p(c | prompt(x))

This only counts more ways of writing the same answer; there is no calibration or prior correction. A softmax
over the label scores gives the soft labels.

## Making it fast

The part of the prompt before the text is run once and its key/value cache is kept. Each text then needs a single
forward pass that covers the text and all labels together: the tokens of all continuations (every surface form of
every label) form a prefix tree, so continuations that start alike, such as `card arrival` and `card linking`,
share nodes, and a 4D attention mask lets each tree node attend only to the prompt, the text and its own
ancestors. There is no padding and no per-label copy of the cache.

When a task has few labels and short texts, several texts are packed into one forward pass the same way, each
attending only to the prompt and itself (`--teacher-tokens`, default 2048 tokens per pass). Only the
log-probabilities of the continuations' own tokens are computed from the outputs, never a softmax over the whole
vocabulary for every position. Tests check the scores against a plain forward pass per continuation, and packed
against unpacked scores.

The scores are cached as JSON lines keyed by a hash of the text, in a file named after a fingerprint of the
teacher, prompt and continuations, so an interrupted run resumes and a changed prompt never reuses old scores.

## Training the student

The student minimises KL(teacher ‖ student) over the training texts, plus `--ce-weight` times the cross-entropy
on any gold labels (`--gold-per-class`, `--gold-file`): standard knowledge distillation (Hinton et al., 2015). For
multi-label tasks it minimises the binary cross-entropy to the teacher's per-label probabilities.

## References

- G. Hinton, O. Vinyals, J. Dean. [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). arXiv:1503.02531, 2015.
- T. Schick, H. Schütze. [Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference](https://arxiv.org/abs/2001.07676). EACL, 2021. (Verbalisers.)
- W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, M. Zhou. [MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957). NeurIPS, 2020.
- N. Reimers, I. Gurevych. [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084). EMNLP, 2019.
- L. Tunstall, N. Reimers, U. E. S. Jo, L. Bates, D. Korat, M. Wasserblat, O. Pereg. [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055). arXiv:2209.11055, 2022. (SetFit.)
