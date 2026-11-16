"""Temporary legacy import shim for the attention migration.

New code imports lightllm.common.attention.backends directly. Removed with the
runtime integration in PR #1721 (5/5), after remaining model callers migrate.
"""

from lightllm.common.attention.backends.mla.triton.kernels.prefill.context_flashattention_nopad_with_v import (
    context_attention_fwd_with_v,
)
