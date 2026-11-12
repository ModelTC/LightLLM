import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("k,n", [(2560, 1280), (640, 2560), (65, 79)])
def test_small_expert_gemm_matches_linear_and_replays_expert_counts(k, n):
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep.bf16 import _expert_gemm

    torch.manual_seed(3107)
    x = torch.randn(512, k, device="cuda", dtype=torch.bfloat16) * 0.2
    w = torch.randn(4, n, k, device="cuda", dtype=x.dtype) * 0.03
    y = torch.full((512, n), 17, device="cuda", dtype=x.dtype)
    ends = torch.tensor([0, 3, 128, 257], device="cuda", dtype=torch.int32)

    def run():
        _expert_gemm(x, w, y, ends, 1, False)

    def check():
        mask = torch.zeros(512, device="cuda", dtype=torch.bool)
        start = 0
        for expert, end in enumerate(ends.tolist()):
            torch.testing.assert_close(y[start:end], F.linear(x[start:end], w[expert]), rtol=0.008, atol=0.002)
            mask[start:end] = True
            start = (end + 127) // 128 * 128
        assert (y[~mask] == 17).all()

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    ends.copy_(torch.tensor([17, 128, 256, 399], device="cuda", dtype=ends.dtype))
    x.mul_(0.5)
    y.fill_(17)
    graph.replay()
    check()


@pytest.mark.parametrize("width", [79, 128, 640])
@pytest.mark.parametrize("kind", ["silu", "gelu", "clamp", "clamp_add_one"])
@pytest.mark.parametrize("shards", [1, 8])
def test_valid_expert_activation_preserves_padding_and_replays_dynamic_counts(width, kind, shards):
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep.bf16 import _activate_valid_rows

    x = torch.randn((512, 2 * width), device="cuda", dtype=torch.bfloat16) * 3
    output = torch.full((512, width), 17.0, device="cuda", dtype=x.dtype)
    ends = torch.tensor([0, 3, 128, 257], device="cuda", dtype=torch.int32)
    clamped = kind.startswith("clamp")

    def run():
        _activate_valid_rows[(4, (width + 127) // 128, shards)](
            x,
            output,
            ends,
            width,
            128,
            1.7 if clamped else None,
            2.0 if clamped else None,
            kind == "clamp_add_one",
            kind == "gelu",
            16,
            128,
            shards,
            num_warps=4,
        )

    def check():
        gate, value = x.chunk(2, -1)
        gate = gate.float()
        if clamped:
            gate = gate.clamp(max=2.0)
            value = value.clamp(-2.0, 2.0)
            gate = gate * torch.sigmoid(gate * 1.7)
            if kind == "clamp_add_one":
                value = value + 1
        elif kind == "gelu":
            gate = F.gelu(gate, approximate="tanh")
        else:
            gate = F.silu(gate)
        expected = (gate.to(x.dtype) * value).to(x.dtype)
        mask = torch.zeros(512, device="cuda", dtype=torch.bool)
        start = 0
        for end in ends.tolist():
            mask[start:end] = True
            start = (end + 127) // 128 * 128
        torch.testing.assert_close(output[mask], expected[mask], rtol=0.008, atol=0.008)
        assert (output[~mask] == 17).all()

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    ends.copy_(torch.tensor([5, 128, 259, 512], device="cuda", dtype=ends.dtype))
    x.mul_(0.5)
    output.fill_(17)
    graph.replay()
    check()
