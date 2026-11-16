"""Temporary legacy import shim for the attention migration.

New code imports lightllm.common.attention.backends directly. Removed with the
runtime integration in PR #1721 (5/5), after remaining model callers migrate.
"""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from lightllm.common.attention.backends.gdn.triton.kernels.chunk import chunk_gated_delta_rule
from lightllm.common.attention.backends.gdn.triton.kernels.fused_recurrent import fused_recurrent_gated_delta_rule

__all__ = [
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
]
