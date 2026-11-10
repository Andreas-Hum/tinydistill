# The method in equations

This page states what tinydistill computes, step by step, as the code does it, and names the function that
implements each step (all under `src/tinydistill/`). For what the steps gain on real tasks, see the
[benchmarks](../results.md); for the same ideas in prose, see [How it works](../how-it-works.md).

| Symbol | Meaning |
|---|---|
| $x$ | an input text |
| $y = 1, \dots, C$ | a label, in the task's order; the task has $C$ labels |
| $\pi(x)$ | the prompt rendered for $x$ (chat template applied), up to where the answer starts |
| $F_y$ | the continuations of label $y$: its surface forms as token sequences, each ending in the end-of-turn token |
| $s_y(x)$ | the teacher's score of label $y$ for $x$ |
| $p(y \mid x)$ | the soft label: the teacher's probability of $y$ |
| $f_\phi(x)$, $q(y \mid x)$ | the student's logits, and its probability of $y$ (the softmax of the logits) |
| $\sigma(t)$ | the logistic sigmoid, $1 / (1 + e^{-t})$ |

## 1. The teacher's label score

The teacher never generates text: it scores continuations. A label's surface forms are its name as written plus
the kinds in the task's `surface_forms` (by default lowercase, Capitalised and UPPERCASE), each with and without
a leading space; forms that tokenise identically count once. Each form is followed by the end-of-turn token (read
from the chat template, else the tokenizer's EOS). For a continuation $c = (c_1, \dots, c_m)$,

$$
\log p(c \mid x) = \sum_{j=1}^{m} \log p_\theta\left(c_j \mid \pi(x), c_1, \dots, c_{j-1}\right)
$$

and a label's score is the log of the summed probability of its forms, computed as a log-sum-exp:

$$
s_y(x) = \log \sum_{c \in F_y} p(c \mid x)
$$

The continuations are built in `teacher.py` by `surface_variants`, `end_of_turn` and `_Prompt._prepare`; the
token log-probabilities come from one forward pass per pack of texts (`LabelScorer._score_pack`) and are summed
per continuation in `LabelScorer._totals`; the log-sum-exp is `_Prompt._label_scores`.

Both are sums, not averages: every token adds a negative term, so a label whose name takes more tokens scores
lower, other things equal ([When to use it](../when-to-use.md)). $s_y$ is not normalised over the labels
either; the softmax in section 4 does that.

**Multi-label tasks** ask one yes/no question per label. $F_y^{+}$ and $F_y^{-}$ are the question's remainder
from `{label}` on (the label's name and the rest of the question), then each surface form of the positive answer
(default `yes`) or the negative one (default `no`), then the end-of-turn token. The score is the log-odds

$$
s_y(x) = \log \sum_{c \in F_y^{+}} p(c \mid x) - \log \sum_{c \in F_y^{-}} p(c \mid x)
$$

where the question's tokens, shared by both sets, cancel (`_Prompt._label_scores`).

Hosted API teachers (`ApiScorer`) score differently and are not covered here; see
[Choosing the teacher](teacher.md).

## 2. The prompt: similar examples and the shortlist

Both use a small sentence encoder (`--few-shot-model` and `--shortlist-model`, default
`sentence-transformers/all-MiniLM-L6-v2`). A text $t$, cut to 128 tokens, gives last-layer states
$h_1, \dots, h_n$ at its $n$ tokens (padding left out), and its embedding is their mean, length-normalised
(`shortlist.py`, `Embedder.embed`):

$$
e(t) = \frac{\bar h}{\lVert \bar h \rVert_2}, \qquad \bar h = \frac{1}{n} \sum_{i=1}^{n} h_i
$$

Similarity is the cosine $e(x)^\top e(t)$.

**Similar examples.** The pool $G$ holds gold (text, label) pairs: with `--gold-per-class K`, those $K$ per
class; with `--few-shot-similar M` alone, every gold-labelled training text. Text $x$ gets $R_M(x)$, the $M$
pairs of $G$ with the highest $e(x)^\top e(t)$, leaving out pairs whose text is $x$ itself, ordered from least
to most similar so that the most similar stands next to $x$ (`Retriever.__call__`). They fill `{examples}` as
lines `- "text" -> label name` (`Task.shots` in `task.py`), so $\pi(x)$ contains $R_M(x)$. With
`--gold-per-class` and `{examples}` in the prompt, $M = 16$ by default. With `--few-shot-sets B`, set
$b = 0, \dots, B-1$ shows the examples at ranks $bM + 1$ to $(b+1)M$, and each set is one member of the
ensemble in section 3.

**Shortlist.** With `--shortlist K`, each label is embedded from `name: description` (or its name), and $x$ is
scored on the set $S(x)$ of the $K$ labels with the highest $e(x)^\top e(\text{label})$, plus the
`--shortlist-keep` labels (`Shortlist.__call__`). The continuations of a label outside $S(x)$ are not run; each
gets log-probability $-10^9$ (`NOT_SCORED`, set in `LabelScorer._totals`), so its probability after the
softmax is zero.

## 3. The probe

Prompt-augmented linear probing (Cho et al., AAAI 2023, [arXiv 2212.10873](https://arxiv.org/abs/2212.10873)).
$h(x) \in \mathbb{R}^d$ is the teacher's state at the last position of $\pi(x)$, the one whose output predicts the
first answer token: by default the input of its LM head (the last layer's output after the final norm), with
`--probe-layer L` the output of decoder layer $L$ there (`LabelScorer._forward`). With `--probe-pool mean` it is
instead the mean over the text's own positions: if $s_1, \dots, s_m$ are those states at the last $m$ positions
of $\pi(x)$, the tokens of the text and of the prompt after it (not the examples before it; at least one),

$$
h(x) = \frac{1}{m} \sum_{j=1}^{m} s_j
$$

(`LabelScorer._pooled`). States are cached as float16.

The probe is fit on the gold texts $(x_i, y_i)$, $i = 1, \dots, N$, after standardising each feature $k$:

$$
\mu = \frac{1}{N} \sum_{i=1}^{N} h(x_i), \qquad
\sigma_k = \max\left(\sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(h_k(x_i) - \mu_k\right)^2},\ 10^{-6}\right), \qquad
z_i = \frac{h(x_i) - \mu}{\sigma}
$$

It is multinomial logistic regression with an L2 penalty toward zero on the weights (not the bias):

$$
\min_{W \in \mathbb{R}^{C \times d},\ b \in \mathbb{R}^{C}}\ \frac{1}{N} \sum_{i=1}^{N} -\log \operatorname{softmax}\left(W z_i + b\right)_{y_i} + \lambda \lVert W \rVert_F^2
$$

with $\lambda$ = `PROBE_L2` $= 10^{-3}$ (`--probe-l2`), solved full-batch in float64 by LBFGS with a strong Wolfe
line search, from $W = 0$, $b = 0$, for at most `PROBE_STEPS` $= 200$ iterations. The scaling is then folded in,
$W' = W \operatorname{diag}(\sigma)^{-1}$ and $b' = b - W'\mu$, and the probe's scores are its log-probabilities

$$
s_y^{\text{probe}}(x) = \log \operatorname{softmax}\left(W' h(x) + b'\right)_y
$$

On a multi-label task the probe is one sigmoid per label, fit with the mean binary cross-entropy on the gold
label sets, and its score is the log-odds $(W' h(x) + b')_y$. See `teacher.py`: `fit_probe`, `probe_scores`,
`ProbeScorer`; `tinydistill label` fits it on all gold texts in `cli.py`, `_probe`.

With gold labels and a local teacher the probe alone is the teacher by default (`--probe-only`): the teacher
only computes $h(x)$ and scores no continuation, and $s = s^{\text{probe}}$. With `--probe` it is one more member
next to the scores of section 1.

**Ensembles.** Members $t = 1, \dots, T$ (teachers, `--prompts`, example sets, the probe) are combined as the log
of their mean probability (`combine_scores`); a single member's scores are kept as they are:

$$
s_y(x) = \log \frac{1}{T} \sum_{t=1}^{T} \operatorname{softmax}\left(s^{(t)}(x)\right)_y
$$

Multi-label, as the log-odds of the mean of each label's probability:
$s_y(x) = \log \frac{1}{T} \sum_t \sigma(s_y^{(t)}) - \log \frac{1}{T} \sum_t \sigma(-s_y^{(t)})$.

## 4. Soft labels and the student's loss

The soft labels are the softmax of the scores, with no temperature and no other change (`cli.py`, `cmd_train`;
`distiller.py`, `Distiller.fit`):

$$
p(y \mid x) = \frac{e^{s_y(x)}}{\sum_{y'=1}^{C} e^{s_{y'}(x)}}
$$

For a single teacher scored as in section 1, this is the probability of a label's forms renormalised over the
continuations of all labels. Multi-label, each label is separate: $p_y(x) = \sigma(s_y(x))$.

**Loss.** For a batch $B$, of which $B_g$ are the texts with a gold label $y^{\ast}(x)$
(`student.py`, `distillation_loss`):

$$
\mathcal{L}(B) = \frac{1}{|B|} \sum_{x \in B} \sum_{y=1}^{C} p(y \mid x) \log \frac{p(y \mid x)}{q(y \mid x)}
\;+\; \lambda_{\mathrm{ce}} \frac{1}{|B_g|} \sum_{x \in B_g} -\log q\left(y^{\ast}(x) \mid x\right)
$$

The first term is the mean KL divergence from the teacher to the student over every text, gold-labelled ones
included; the second is dropped when $B_g$ is empty. $\lambda_{\mathrm{ce}}$ is `--ce-weight` (default 1 in
`tinydistill train`). Multi-label (`distillation_loss_multilabel`), with $z = f_\phi(x)$ and
$\mathrm{BCE}(a, t) = -t \log a - (1 - t) \log(1 - a)$, averaged over labels and texts:

$$
\mathcal{L}(B) = \frac{1}{|B|\,C} \sum_{x \in B} \sum_{y=1}^{C} \mathrm{BCE}\left(\sigma(z_y), p_y(x)\right)
+ \lambda_{\mathrm{ce}} \frac{1}{|B_g|\,C} \sum_{x \in B_g} \sum_{y=1}^{C} \mathrm{BCE}\left(\sigma(z_y), y^{\ast}_y(x)\right)
$$

with $y^{\ast}(x)$ the multi-hot gold labels.

**Optimisation** (`train_student`, `_schedule`): AdamW with learning rate $10^{-4}$ (`--lr`) and weight decay
0.01, linear warm-up over the first tenth of the steps then linear decay to zero, gradient norm clipped at 1,
batches of 32 (`--batch-size`), 10 epochs (`--epochs`), texts cut to 128 tokens (`--max-length`).
`--warmup-pairs` (a contrastive step) and `--init-head names` act before distillation and leave
$\mathcal{L}$ unchanged; see [Training the student](training.md).

**Held-out texts and early stopping.** With `--val-share S` above 0, $\max(1, \operatorname{round}(S \cdot N))$ of
the $N$ training texts, drawn from those without a gold label, are held out of training. After each epoch $e$ the
mean of $\mathcal{L}$ over them is measured (the teacher term only, as they have no gold label); the epoch $e^{\ast}$
with the lowest one is kept, and training stops once $e - e^{\ast} \geq P$, the patience (`--patience`, default 3).
The default $S = 0$ runs every epoch and keeps the last.

**The student.** A pretrained encoder loaded with `transformers`' `AutoModelForSequenceClassification`, with a
new head of $C$ outputs; each encoder brings its own head. For the default `sentence-transformers/all-MiniLM-L6-v2`
(a BERT model) that is the `[CLS]` token's last-layer state through BERT's pooler, then a linear layer
(with dropout on the pooled vector while training):

$$
f_\phi(x) = W_c \tanh\left(W_p h_{[\mathrm{CLS}]}(x) + b_p\right) + b_c
$$

The static student (`--student static`, `StaticForSequenceClassification` in `static.py`) averages one learned
vector $E_t$ per token over the text's $n$ tokens, then applies a linear layer:
$f_\phi(x) = W \frac{1}{n} \sum_{i=1}^{n} E_{t_i} + b$.

## 5. int8 export

`export.py`, `quantize_int8` calls onnxruntime's `quantize_dynamic(..., weight_type=QInt8, per_channel=True)` and
leaves its other options at their defaults: no reduced range, symmetric weights (the default for a signed type),
and only MatMuls whose second input is a constant weight. Every linear layer, the pooler and the classifier
included, becomes such a MatMul with a weight $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$, quantised
with one scale per output channel $j$ and zero point 0:

$$
\alpha_j = \frac{\max_i \lvert W_{ij} \rvert}{127}, \qquad
Q_{ij} = \operatorname{round}\left(\frac{W_{ij}}{\alpha_j}\right) \in [-127, 127], \qquad
\hat W_{ij} = \alpha_j Q_{ij}
$$

The activations are quantised at run time: the input $X$ of each such MatMul goes through ONNX's
[DynamicQuantizeLinear](https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html), to uint8 with one
scale $\beta$ and zero point $z_X$ for the whole tensor (every text and position in the batch):

$$
\beta = \frac{\max(\max X, 0) - \min(\min X, 0)}{255}, \qquad
z_X = \operatorname{round}\left(\frac{-\min(\min X, 0)}{\beta}\right), \qquad
X^{q} = \operatorname{clip}\left(\operatorname{round}(X / \beta) + z_X,\ 0,\ 255\right)
$$

`MatMulInteger` multiplies $X^{q} - z_X$ by $Q$ in integers, and the result is scaled back by $\beta \alpha_j$.
The products of two activations inside attention stay in float32, and the embedding tables are stored as uint8
with one scale and zero point per table. Because $\beta$ depends on the whole batch, `Predictor.logits` runs the
int8 model one text at a time by default.

## 6. The cascade

The student's confidence is its top probability, $c(x) = \max_y q(y \mid x)$. A text with $c(x) < \tau$ goes to
the teacher, which answers $\arg\max_y s_y(x)$ with the run's teacher as it labelled; every other text keeps the
student's $\arg\max_y q(y \mid x)$. In use, $\tau$ is `--threshold` of `predict --cascade` and
`serve --cascade` (default 0.5); see `export.py`, `Cascade.predict`. The cascade is for single-label tasks.

The results tables (`evaluate.py`, `cascade_curve`) sort the $n$ test texts' confidences,
$c_{(1)} \leq \dots \leq c_{(n)}$. For a target share $s$, with $k = \operatorname{round}(s n)$, the threshold is
$\tau = c_{(k+1)}$ ($\tau = 1$ when $k = n$), and the texts with $c(x) < \tau$ go to the teacher. Ties at $\tau$
stay with the student, so the share actually sent, $s'$ (the share of texts with $c(x) < \tau$), is at most
$s$. Accuracy is that of the combined answers, and the latency per text (`cascade_table`) is

$$
t = t_S + s' \, t_T
$$

with $t_S$ and $t_T$ the batch-1 latencies per text of the student and the teacher: the student reads every
text, and the teacher the share sent to it. The thresholds are set on the test texts to hit each share; in
deployment, set $\tau$ on a validation split.
