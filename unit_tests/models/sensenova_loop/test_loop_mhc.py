import pytest
import torch
import torch.nn.functional as F

from lightllm.models.sensenova_loop.triton_kernel.mhc import hc_contract, hc_expand, hc_post, hc_pre_norm
from lightllm.models.sensenova_loop.triton_kernel.mhc.pre_norm import has_deepgemm_mhc
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _route_reference(x, weight, bias, streams, iterations):
    # Training normalizes the residual before the FP32 router projection.
    flat = x.float()
    logits = F.linear(flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6), weight, bias)
    pre = logits[:, :streams].sigmoid()
    post = 2 * logits[:, streams : 2 * streams].sigmoid()
    logits = logits[:, 2 * streams :].reshape(-1, streams, streams)
    for _ in range(iterations):
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
    mix = logits.exp()
    branch = (pre.to(x.dtype).unsqueeze(1) @ x.view(-1, streams, x.shape[1] // streams)).squeeze(1)
    return branch, mix, post


@pytest.mark.parametrize("tokens", [1, 19, 257])
@pytest.mark.parametrize("streams", [2, 4])
def test_loop_mhc_matches_training_routing_and_rounding(tokens, streams):
    torch.manual_seed(35)
    hidden = 4096
    x = torch.randn(tokens, streams * hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2 * streams + streams * streams, streams * hidden, device="cuda") * 0.008
    bias = torch.randn(weight.shape[0], device="cuda")
    norm = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    expected = _route_reference(x, weight, bias, streams, 20)
    actual = hc_pre_norm(
        x,
        weight,
        bias,
        norm,
        streams,
        1e-6,
        1e-6,
        20,
        use_deepgemm=False,
    )
    torch.testing.assert_close(actual[0], rmsnorm_forward(expected[0], norm, 1e-6), atol=0.02, rtol=0.02)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual[2], expected[2], atol=2e-6, rtol=2e-5)
    layer_out = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    # Qwen uses output,input order; both matmul and the post product round to bf16.
    expected_out = expected[1].to(x.dtype) @ x.view(tokens, streams, hidden)
    expected_out += expected[2].to(x.dtype).unsqueeze(-1) * layer_out.unsqueeze(1)
    actual_out = hc_post(layer_out, x, actual[1].transpose(1, 2), actual[2], streams)
    torch.testing.assert_close(actual_out.view_as(expected_out), expected_out, atol=0.02, rtol=0.01)


@pytest.mark.skipif(not has_deepgemm_mhc(), reason="DeepGEMM mHC projection unavailable")
def test_loop_deepgemm_projection_matches_torch():
    x = torch.randn(19, 16384, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(24, 16384, device="cuda") * 0.005
    bias = torch.randn(24, device="cuda")
    norm = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    actual = hc_pre_norm(x, weight, bias, norm, 4, 1e-6, 1e-6, 20)
    expected = hc_pre_norm(x, weight, bias, norm, 4, 1e-6, 1e-6, 20, use_deepgemm=False)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("streams", [2, 4])
def test_loop_mhc_decode_can_capture_and_replay_cuda_graph(streams):
    x = torch.randn(3, streams * 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2 * streams + streams * streams, streams * 4096, device="cuda") * 0.005
    bias = torch.randn(weight.shape[0], device="cuda")
    norm = torch.ones(4096, device="cuda", dtype=torch.bfloat16)

    def forward():
        branch, mix, post = hc_pre_norm(
            x,
            weight,
            bias,
            norm,
            streams,
            1e-6,
            1e-6,
            20,
            use_deepgemm=False,
        )
        return hc_post(branch, x, mix.transpose(1, 2), post, streams)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    x.normal_()
    expected = forward()
    graph.replay()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    collapsed = torch.randn(3, 4096, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(hc_contract(hc_expand(collapsed, streams), streams), collapsed)
