import inspect

import pytest
import torch

from lightllm.common.basemodel.attention.linear.flashinfer import chunk_gated_delta_rule
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops import chunk_gated_delta_rule as native


@pytest.fixture(autouse=True)
def require_flashinfer_gdn():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FlashInfer GDN regression requires Hopper")
    module = pytest.importorskip("flashinfer.gdn_prefill")
    if "use_cp" not in inspect.signature(module.chunk_gated_delta_rule).parameters:
        pytest.skip("FlashInfer >= 0.7.2 required")


@pytest.mark.parametrize("lengths", [[17, 31], [512], [2048], [8192], [0, 19, 0, 47], [1024] * 8])
def test_prefill_nonzero_state_and_graph(lengths):
    torch.manual_seed(4317)
    tokens = sum(lengths)
    packed = torch.randn(1, tokens, 10, 128, device="cuda", dtype=torch.bfloat16)
    q, k, v = packed[:, :, :2], packed[:, :, 2:4], packed[:, :, 4:]
    v.mul_(0.2)
    g = -torch.rand(1, tokens, 6, device="cuda") * 0.1
    beta = torch.rand_like(g)
    state = torch.randn(len(lengths), 6, 128, 128, device="cuda") * 0.01
    cu = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=True,
    )
    original = state.clone()
    if 0 in lengths:
        active = [i for i, length in enumerate(lengths) if length]
        offsets = torch.tensor([0, *[lengths[i] for i in active]], device="cuda", dtype=torch.int32).cumsum(
            0, dtype=torch.int32
        )
        expected, active_state = native(**dict(kwargs, initial_state=state[active], cu_seqlens=offsets))
        expected_state = state.clone()
        expected_state[active] = active_state
    else:
        expected, expected_state = native(**kwargs)
    actual, actual_state = chunk_gated_delta_rule(**kwargs)
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.03)
    torch.testing.assert_close(actual_state, expected_state, atol=0.003, rtol=0.03)
    torch.testing.assert_close(state, original, atol=0, rtol=0)
    output_only, final = chunk_gated_delta_rule(**dict(kwargs, output_final_state=False))
    assert final is None
    torch.testing.assert_close(output_only, actual, atol=0, rtol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, final = chunk_gated_delta_rule(**kwargs)
    try:
        for _ in range(3):
            packed.mul_(0.7)
            state.add_(0.001)
            cu.copy_(torch.tensor([0, *reversed(lengths)], device="cuda").cumsum(0))
            graph.replay()
            expected, expected_state = chunk_gated_delta_rule(**kwargs)
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
            torch.testing.assert_close(final, expected_state, atol=0, rtol=0)
    finally:
        graph.reset()


def test_prefill_continuation_uses_native_state_layout():
    torch.manual_seed(801)
    q = torch.randn(1, 768, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, 768, 6, 128, device="cuda", dtype=torch.bfloat16) * 0.2
    g = -torch.rand(1, 768, 6, device="cuda") * 0.1
    beta = torch.rand_like(g)
    initial = torch.randn(1, 6, 128, 128, device="cuda") * 0.01

    def run(kernel, start, end, state):
        return kernel(
            q=q[:, start:end],
            k=k[:, start:end],
            v=v[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            initial_state=state,
            output_final_state=True,
            cu_seqlens=torch.tensor([0, end - start], device="cuda", dtype=torch.int32),
            use_qk_l2norm_in_kernel=True,
        )

    _, prefix = run(native, 0, 256, initial)
    output, final = run(chunk_gated_delta_rule, 256, 768, prefix)
    expected, expected_state = run(native, 256, 768, prefix)
    torch.testing.assert_close(output, expected, atol=0.001, rtol=0.03)
    torch.testing.assert_close(final, expected_state, atol=0.003, rtol=0.03)
    expected, expected_state = run(native, 0, 768, initial)
    torch.testing.assert_close(output, expected[:, 256:], atol=0.001, rtol=0.03)
    torch.testing.assert_close(final, expected_state, atol=0.003, rtol=0.03)
