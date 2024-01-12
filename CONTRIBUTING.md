## Contributing Guidelines

### Coding Style Guide

In general, we adhere to [Google Python style guide](https://google.github.io/styleguide/pyguide.html), and we recommend to use `yapf` to format your code.

In this project, we adopted `pre-commit` to automatic check the code style.

To begin with, you should follow the step below to install `pre-commit`.

```bash
pip install pre-commit
```

Then, you should config the pre-commit hook as below.

```bash
pre-commit install
```

Then when you commit your change, your code will be automatically checked.
