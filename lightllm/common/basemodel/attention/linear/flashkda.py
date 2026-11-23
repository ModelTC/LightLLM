# SPDX-License-Identifier: Apache-2.0

"""Choose the KDA prefill kernel from the server arguments: FlashKDA when requested or usable, else Triton."""

import torch

from lightllm.common.attention.backends.kda.flashkda import load_flash_kda_fwd


def get_flash_kda_fwd(model):
    backends = model.args.llm_prefill_att_backend
    requested = backends[1] if len(backends) > 1 else "auto"
    ssm_dtype = getattr(torch, model.args.linear_att_ssm_data_type)
    return load_flash_kda_fwd(requested, dtype=model.data_type, ssm_dtype=ssm_dtype)
