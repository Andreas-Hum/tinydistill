# Contributing to tinydistill

Thanks for your interest. Bug reports, documentation fixes, new examples and code are all welcome. By taking
part you agree to follow the [code of conduct](https://github.com/Andreas-Hum/tinydistill/blob/main/CODE_OF_CONDUCT.md).

## Reporting bugs and asking for features

Open an issue with the matching form. For a bug, include the `tinydistill` version, the command you ran, the
task file (or a shortened one) and the full error. Security problems go through the private process in
[SECURITY.md](https://github.com/Andreas-Hum/tinydistill/blob/main/SECURITY.md), not a public issue.

## Development setup

Everything, including the tests, runs on a CPU:

```sh
git clone https://github.com/Andreas-Hum/tinydistill && cd tinydistill
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest
```

The tests use tiny random models from `hf-internal-testing`, so they need a network connection the first time
and then take under a minute.

## Pull requests

- Keep a pull request to one change, and describe what it changes and why.
- Add or update tests. New scoring code is checked against a plain forward pass per continuation (see
  `tests/test_scoring.py`); new losses and metrics against values worked out by hand.
- Run `ruff check .`, `ruff format --check .` and `pytest` before pushing. CI runs the tests on Linux
  (Python 3.11 to 3.13), macOS and Windows, on the oldest supported dependency versions, against the built
  wheel, and opens a `tinydistill web` page in Chromium (`tests/browser/`); it also runs weekly against the
  newest dependency releases. A release runs all of it before publishing.
- Update the README, the documentation in `docs/` and `CHANGELOG.md` when behaviour or options change. To
  preview the documentation: `pip install -e ".[docs]"` and `mkdocs serve`. The command-line reference and the
  Python API pages are generated from the code and its docstrings.
- Numbers in the README come only from saved result files (`examples/*/*.json`, written by the example
  scripts); never type a result by hand.

## Scope

tinydistill does standard knowledge distillation (Hinton et al., 2015): the teacher's label scores are
summed log-probabilities over each label's surface forms, the soft labels are their softmax, and the student
minimises KL(teacher || student), optionally with cross-entropy on gold labels. Changes that alter the
teacher's probabilities after scoring, such as calibration, prior correction or class reweighting, are out of
scope for this project; please open an issue to discuss before working on anything in that direction.

## Releases

Maintainers release by publishing a GitHub release tagged `vX.Y.Z` that matches the version in
`pyproject.toml`; the release workflow builds the package and uploads it to PyPI.
