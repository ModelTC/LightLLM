import pytest
import torch
import torch.nn.functional as F

from lightllm.common.basemodel.triton_kernel.linear_att import mtp_fused_recurrent as kernel_module
from unit_tests.common.basemodel.triton_kernel.linear_att.test_mtp_fused_recurrent_autotune import (
    autotune_environment,
    make_inputs,
    rebuild,
)


def reference(inputs, round_intermediates):
    state = inputs["initial_state"].clone()
    q, k, v = [inputs[name][0].float() for name in ("q", "k", "v")]
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    beta = inputs["b_raw"].float().sigmoid()
    if round_intermediates:
        q = q.to(inputs["q"].dtype).float()
        k = k.to(inputs["k"].dtype).float()
        beta = beta.to(inputs["b_raw"].dtype).float()
    group_size = v.shape[1] // q.shape[1]
    q = q.repeat_interleave(group_size, dim=1) * q.shape[-1] ** -0.5
    k = k.repeat_interleave(group_size, dim=1)
    g = -inputs["A_log"].float().exp() * F.softplus(inputs["a_raw"].float() + inputs["dt_bias"].float())
    output = torch.zeros_like(inputs["v"], dtype=torch.float32)
    cumulative = inputs["cu_seqlens"].tolist()
    for row, (start, end) in enumerate(zip(cumulative, cumulative[1:])):
        read_slot = inputs["ssm_state_indices"][row, inputs["num_accepted_tokens"][row] - 1]
        h = inputs["initial_state"][read_slot].float().clone()
        for token in range(start, end):
            h *= g[token].exp()[:, None, None]
            delta = (v[token] - (h * k[token, :, :, None]).sum(1)) * beta[token, :, None]
            h += k[token, :, :, None] * delta[:, None, :]
            output[0, token] = (h * q[token, :, :, None]).sum(1)
            slot = inputs["ssm_state_write_indices"][row, token - start]
            state[slot] = h.to(state.dtype)
            h = state[slot].float().clone()
    return output, state


def test_intermediate_precision_separates_tuning_and_survives_rebuild():
    inputs = make_inputs()
    kernel = kernel_module.mtp_fused_recurrent_gated_delta_rule
    original = kernel._static_key(**inputs)
    assert "round_intermediates" not in original
    assert kernel._static_key(**inputs, round_intermediates=False) == original
    inputs["round_intermediates"] = True
    rounded = kernel._static_key(**inputs)
    assert rounded != original
    rebuilt = rebuild(inputs)
    assert rebuilt["round_intermediates"] is True
    assert kernel._static_key(**rebuilt) == rounded


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("round_intermediates", [False, True])
def test_intermediate_precision_and_persistent_state(dtype, state_dtype, round_intermediates):
    inputs = make_inputs(lengths=(3, 1, 0), device="cuda", dtype=dtype, state_dtype=state_dtype, separate_write=True)
    initial = inputs["initial_state"].clone()
    expected_output, expected_state = reference(inputs, round_intermediates)
    kernel = kernel_module.mtp_fused_recurrent_gated_delta_rule
    for config in ({"BV": 8, "num_warps": 1, "num_stages": 1}, {"BV": 32, "num_warps": 4, "num_stages": 2}):
        inputs["initial_state"].copy_(initial)
        output, state = kernel(**inputs, round_intermediates=round_intermediates, run_config=config)
        assert state is inputs["initial_state"]
        torch.testing.assert_close(output.float(), expected_output, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(state.float(), expected_state.float(), atol=2e-3, rtol=2e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, state = kernel(**inputs, round_intermediates=round_intermediates)
    for changed in (False, True):
        if changed:
            inputs["q"].neg_()
            inputs["b_raw"].add_(0.5)
            inputs["num_accepted_tokens"].fill_(1)
        inputs["initial_state"].copy_(initial)
        expected_output, expected_state = reference(inputs, round_intermediates)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output.float(), expected_output, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(state.float(), expected_state.float(), atol=2e-3, rtol=2e-2)
