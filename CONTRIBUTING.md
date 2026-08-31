## Contributing Guidelines

### Coding Style Guide

This project uses [`black`](https://github.com/psf/black) (line length 120) and `flake8` for code style. Both are run automatically via `pre-commit` (see `.pre-commit-config.yaml` for the exact configuration).

To begin with, you should follow the step below to install `pre-commit`.

```bash
pip install pre-commit
```

Then, you should config the pre-commit hook as below.

```bash
pre-commit install
```

Then when you commit your change, your code will be automatically checked.
