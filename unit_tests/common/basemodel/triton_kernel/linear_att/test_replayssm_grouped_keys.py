"""The storage optimization must be bit-identical to the redundant-key layout."""

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMBackend
from lightllm.common.basemodel.triton_kernel.linear_att.compact_ssm import CompactSSMBackend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("mode", ["replay", "compact"])
@pytest.mark.parametrize("group", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_keys_graph_fold_partial_accept(mode, group, dtype):
    torch.manual_seed(192)
    layers, slots, h, kd, vd, width = 2, 5, 2, 128, 128, 4
    hv = h * group
    state = torch.randn(layers, slots, hv, kd, vd, device="cuda", dtype=dtype) * 0.01

    def make_ssm_backend(key_heads):
        if mode == "replay":
            return ReplaySSMBackend(state.clone(), 16, width, num_key_heads=key_heads)
        return CompactSSMBackend(state.clone(), width, torch.bfloat16, num_key_heads=key_heads)

    ssm_backends = [make_ssm_backend(hv), make_ssm_backend(h)]
    assert ssm_backends[0].keys.numel() == ssm_backends[1].keys.numel() * group
    # Packed strides, out-of-order requests, an empty real request, nonempty HOLD.
    packed = torch.randn(1, 8, (2 * h + hv) * kd, device="cuda", dtype=torch.bfloat16)
    q, k, v = [x.view(1, 8, heads, kd) for x, heads in zip(packed.split([h * kd, h * kd, hv * kd], -1), [h, h, hv])]
    a = torch.randn(8, hv, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8, hv, device="cuda", dtype=torch.bfloat16)
    alog = torch.randn(hv, device="cuda") * 0.1
    bias = torch.randn(a.shape[-1], device="cuda") * 0.1
    reqs = torch.tensor([2, 0, 1, 4], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 4, 7, 7, 8], device="cuda", dtype=torch.int32)
    accepted = torch.tensor([0, -1, 0, -1, -1], device="cuda", dtype=torch.int32)

    def step(ssm_backend):
        ssm_backend.prepare_decode(reqs, cu)
        out = [ssm_backend.forward(layer, q, k, v, a, b, alog, bias, reqs, cu)[0] for layer in range(layers)]
        ssm_backend.accept_updates(reqs, accepted)
        return out

    graphs, outputs = [], []
    for ssm_backend in ssm_backends:
        step(ssm_backend)
        ssm_backend.state.copy_(state)
        if mode == "replay":
            ssm_backend.cursors.zero_()
        reqs.fill_(4)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs.append(step(ssm_backend))
        graphs.append(graph)
        reqs.copy_(torch.tensor([2, 0, 1, 4], device="cuda", dtype=torch.int32))

    for iteration in range(67):
        packed.normal_()
        a.normal_().sub_(3)
        b.normal_()
        accepted[0] = iteration % 3
        accepted[2] = (iteration + 1) % 4
        for graph in graphs:
            graph.replay()
        for left, right in zip(*outputs):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
            assert torch.count_nonzero(right[:, -1]) == 0
        if iteration in (18, 39, 66):
            for ssm_backend in ssm_backends:
                ssm_backend.merge_accepted_updates(reqs)
        torch.testing.assert_close(ssm_backends[0].state, ssm_backends[1].state, atol=0, rtol=0)
        torch.testing.assert_close(ssm_backends[1].state[:, [1, 3, 4]], state[:, [1, 3, 4]], atol=0, rtol=0)
        if mode == "replay":
            torch.testing.assert_close(ssm_backends[0].cursors, ssm_backends[1].cursors, atol=0, rtol=0)
        if iteration == 40:
            # Request-slot reuse / checkpoint restore must not resurrect history.
            for ssm_backend in ssm_backends:
                ssm_backend.clear_history(2)
                ssm_backend.state[:, 2].copy_(state[:, 2])
