# When to use it

**Use tinydistill** when an LLM prompt already classifies your texts well enough, but it is too slow, too expensive
or too large to run on every text: the student keeps most of its quality in a model that runs in about a
millisecond on a CPU. It also fits when you have plenty of unlabelled texts and few or no labels.

**Do not use it** when the LLM itself is not good at the task (the student inherits its mistakes), when you have
enough gold labels to train a classifier directly, or when the task needs generated text rather than a choice
among labels. With only a few labels and no GPU, [SetFit](https://github.com/huggingface/setfit) is a good
alternative: on Banking77, with the same 616 labels, it reached 0.797 against tinydistill's 0.813, trains in a
minute and needs no LLM ([Benchmarks](results.md#compared-with-setfit)).

## Limitations

- **The teacher caps the student.** The student learns to imitate the teacher, mistakes included. A small LLM
  choosing zero-shot among many fine-grained labels can be weak; a bigger teacher (try `--load-4bit` with a 7B
  model) or a better task file usually helps more than anything on the student side.
- **A too-small teacher can collapse** onto one or two labels, and so can a capable one on an unlucky task file:
  on AG News, label descriptions made `Qwen2.5-1.5B-Instruct` answer "world" for nearly every article. Check the
  task file with [`tune`](guide/improving.md#check-the-setup-first-tune) before labelling.
- **Scores are sums, not averages.** Longer label names get lower scores, so very uneven name lengths bias the
  teacher. Keep names short and of similar length.
- **Only case and spacing variants are counted.** Plurals, synonyms or rephrasings of a label are not; keep label
  names close to how the teacher would answer.
- **Few-shot examples can hurt** on skewed tasks: they tell the teacher that the classes are equally common.
  Compare zero-shot and few-shot before relying on it.
- **Teacher cost grows with labels.** Use `--shortlist K` with very many labels; a gold label outside the
  shortlist cannot be learned from the teacher for that text.
- **Multi-label tasks** ask one yes/no question per label (all of a text's questions in one forward pass, but a
  longer one), and their students need more epochs.
- **Inputs are truncated** to 128 tokens by default (`--max-length`).
- **Timings depend on the machine.** Every results table states the threads and devices used; rerun `eval` on your
  own hardware.
