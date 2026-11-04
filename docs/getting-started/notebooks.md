# Notebooks

Three notebooks run on a free GPU in Google Colab, without installing anything. Choose Runtime > Change
runtime type > T4 GPU after opening one.

| Notebook | What it shows | |
|---|---|---|
| Quickstart | distil a sentiment classifier in about ten minutes, with charts of what the student learned | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Andreas-Hum/tinydistill/blob/main/notebooks/quickstart.ipynb) |
| Your own task | write a task for your labels, let `tune` choose the task file, distil on your texts, find the mistakes, download the model | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Andreas-Hum/tinydistill/blob/main/notebooks/your_task.ipynb) |
| Deploy | the model from Python, behind an HTTP server, in a web page, and on the Hugging Face Hub | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Andreas-Hum/tinydistill/blob/main/notebooks/deploy.ipynb) |

They also run in Jupyter on your own machine; the files are in
[`notebooks/`](https://github.com/Andreas-Hum/tinydistill/tree/main/notebooks).
