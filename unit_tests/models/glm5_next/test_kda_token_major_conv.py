# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d import causal_conv1d_fn
from lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d_token_major import (
    causal_conv1d_prefill_token_major,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires Hopper CUDA",
)
@pytest.mark.parametrize("width", [3, 4])
def test_token_major_conv_preserves_output_and_history(width):
    torch.manual_seed(42)
    # Include a one-tile warm suffix, a long cold request, and a pad slot.
    lengths = (2049, 2, 1024)
    opts = dict(dtype=torch.bfloat16, device="cuda")
    projection = torch.randn(sum(lengths), 24896, **opts)
    x = projection[:, :24576]
    weight = torch.randn(24576, 4, **opts) * 0.125
    states = torch.randn(4, 24576, width, **opts)
    starts = torch.tensor([0, lengths[0], sum(lengths[:2]), sum(lengths)], dtype=torch.int32, device="cuda")
    slots = torch.tensor([1, 2, -1], dtype=torch.int32, device="cuda")
    warm = torch.tensor([False, True, False], device="cuda")
    reference_states = states.clone()
    expected = causal_conv1d_fn(
        x.transpose(0, 1),
        weight,
        query_start_loc=starts,
        cache_indices=slots,
        has_initial_state=warm,
        conv_states=reference_states[..., :3],
    ).transpose(0, 1)
    actual = causal_conv1d_prefill_token_major(x, weight, states[..., :3], starts, slots, warm, max(lengths))

    assert torch.equal(actual.view(torch.uint8), expected.contiguous().view(torch.uint8))
    assert torch.equal(states.view(torch.uint8), reference_states.view(torch.uint8))
