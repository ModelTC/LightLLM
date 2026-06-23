# Vendored SGLang JIT Subset

This directory contains the minimal SGLang JIT source subset needed by the
DeepSeek-V4 LightLLM implementation.

Source: https://github.com/sgl-project/sglang
Commit: 8cea0473ea5299bc04885f8f6ba71269415a39b5
License: Apache License 2.0, copied in `LICENSE`.

Local changes:
- The Python imports were moved from `sglang.jit_kernel.*` to
  `lightllm.third_party.sglang_jit.*`.
- The package exports only the DSv4 functions used by LightLLM.
