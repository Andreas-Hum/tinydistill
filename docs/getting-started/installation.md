# Installation

```sh
pip install tinydistill               # label, train, export, predict
pip install "tinydistill[hf]"         # + Hugging Face datasets as input (hf:<dataset>:<split>)
pip install "tinydistill[hf,4bit]"    # + 4-bit teachers with bitsandbytes (CUDA)
pip install "tinydistill[plot]"       # + charts of a run (matplotlib)
```

tinydistill needs Python 3.11 or newer and runs on Linux, macOS and Windows; the test suite runs on all three.

## GPU or CPU?

| Step | Where it runs well |
|---|---|
| Labelling with the teacher | a GPU for 1.5B models and larger; a CPU is fine for a few hundred texts with a small teacher |
| Training the student | a GPU is faster, a CPU works |
| Inference with the student | a CPU: about a millisecond per text for the int8 model |

For a CPU-only PyTorch, which is much smaller to download, install it first:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "tinydistill[hf]"
```

No GPU at all? The [Colab notebooks](notebooks.md) run on a free GPU in the browser.

## Hugging Face access

Models and datasets download from the Hugging Face Hub without an account. If downloads are slow or you see a
warning about unauthenticated requests, log in once with a token from
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access is enough; write access
only for `tinydistill push`):

```sh
hf auth login
```
