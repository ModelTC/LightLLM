# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import triton
import triton.language as tl


@triton.jit
def div_normal(x, y):
    return x / y


div = div_normal
exp = tl.exp
log = tl.log
log2 = tl.log2


if not hasattr(tl, "gather"):

    @triton.jit
    def gather(src, index, axis, _builder=None):
        # This is a fallback implementation when tl.gather is not supported
        # In order to pass triton compiler, there is no actual gather operation
        return src

else:
    gather = tl.gather
