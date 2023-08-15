---
name: Bug report
about: Create a report to help us improve
title: "[BUG]"
labels: bug
assignees: ''

---

**Before you submit an issue, please search for existing issues to avoid duplicates.**

**Issue description:**

Please provide a clear and concise description of your issue.

**Steps to reproduce:**

Please list the steps to reproduce the issue, such as:

1. `command 0`
2. `command 2`
3. `command 3`
4. See error

**Expected behavior:**

Please describe what you expected to happen.

**Error logging:**

If applicable, please copy and paste the error message or stack trace here. Use code blocks for better readability.

**Environment:**

Please provide information about your environment, such as:

- [ ] Using container

- OS: (Ubuntu 14.04, CentOS7)
- GPU info:
  - `nvidia-smi` (e.g. `NVIDIA-SMI 525.116.04   Driver Version: 525.116.04   CUDA Version: 12.0`)
  - Graphics cards: (e.g. 4090x8)
- Python: (e.g. CPython3.9)
  - currently, only python>=3.9 is supported
- LightLLm: (git commit-hash)
  - for container: `docker run --entrypoint cat --rm ghcr.io/modeltc/lightllm:main /lightllm/.git/refs/heads/main`
- openai-triton: `pip show triton`

**Additional context:**

Please add any other context or screenshots about the issue here.

**Language:**

Please use English as much as possible for better communication.
