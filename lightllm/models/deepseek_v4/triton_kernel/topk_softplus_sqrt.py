# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the LightLLM project
#
# Wraps the CUDA router adapted from vLLM commit
# 36d2a5086bae12d0c0b311607373e4fc2d1036aa.

import functools
import hashlib
import os

import torch


@functools.lru_cache(maxsize=1)
def _load_cuda():
    from torch.utils.cpp_extension import load

    src = os.path.join(os.path.dirname(__file__), "csrc", "topk_softplus_sqrt.cu")
    flags = ["-O3"]
    with open(src, "rb") as source_file:
        source = source_file.read()
    capability = torch.cuda.get_device_capability()
    cache_key = b"\0".join(
        [
            source,
            " ".join(flags).encode(),
            torch.__version__.encode(),
            str(torch.version.cuda).encode(),
            f"sm{capability[0]}{capability[1]}".encode(),
            os.environ.get("TORCH_CUDA_ARCH_LIST", "").encode(),
        ]
    )
    module_name = f"lightllm_dsv4_router_v1_{hashlib.sha256(cache_key).hexdigest()[:16]}"
    return load(
        name=module_name,
        sources=[src],
        extra_cuda_cflags=flags,
        verbose=False,
    )


@torch.no_grad()
def topk_softplus_sqrt(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    logits: torch.Tensor,
    routed_scaling_factor: float,
    correction_bias: torch.Tensor = None,
    input_ids: torch.Tensor = None,
    tid2eid: torch.Tensor = None,
    bias_vl: torch.Tensor = None,
    image_token_start: int = 0,
) -> None:
    _load_cuda().topk_softplus_sqrt(
        topk_weights,
        topk_indices,
        logits,
        float(routed_scaling_factor),
        correction_bias,
        input_ids,
        tid2eid,
        bias_vl,
        int(image_token_start),
    )
    return
