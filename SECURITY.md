# Security policy

## Supported versions

Security fixes go into the latest release on PyPI. Please upgrade before reporting.

## Reporting a vulnerability

Report it privately through GitHub: on the repository's **Security** tab, choose **Report a vulnerability**.
Please do not open a public issue. Include what is affected, how to reproduce it and what an attacker could do
with it. You will get a reply within a week; a fix and a release follow as soon as practical, and you will be
credited unless you prefer otherwise.

## What to keep in mind when using tinydistill

- **Models and datasets come from the Hugging Face Hub.** tinydistill does not pass `trust_remote_code`, so it
  does not run code from model repositories, but you are still trusting the publishers of the models and data
  you name.
- **`tinydistill serve` has no authentication.** It listens on 127.0.0.1 by default; put it behind your own
  authentication and TLS before exposing it with `--host`.
- **`--vllm-url` sends your texts to that server.** Use a server you control.
- **`tinydistill push` uploads the model and its results to the Hub.** Check that the model card and results
  contain nothing private before pushing to a public repository.
