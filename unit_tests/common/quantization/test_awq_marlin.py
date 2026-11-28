import pytest
import torch

import lightllm.common.basemodel
from lightllm.common.quantization.awq_marlin import can_prepare_awq, dense_awq_marlin, moe_awq_marlin, prepare_awq
from lightllm.common.quantization.quantize_method import WeightPack


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "height,widths,tokens",
    [
        (256, (192, 64, 64), 17),
        (256, (512, 256, 256), 128),
        (256, (512, 256, 256), 129),
        (896, (1024, 256, 256), 16),
        (896, (1024, 256, 256), 128),
        (896, (1024, 256, 256), 129),
    ],
)
def test_marlin_repacking_merged_awq_projection(dtype, height, widths, tokens):
    torch.manual_seed(12)
    width = sum(widths)
    packed = torch.randint(-(2 ** 31), 2 ** 31 - 1, (height, width // 8), dtype=torch.int32, device="cuda")
    zeros = torch.randint_like(packed[: height // 128], -(2 ** 31), 2 ** 31 - 1)
    scales = torch.rand(height // 128, width, dtype=dtype, device="cuda") / 64
    pack = WeightPack(packed, scales, zeros)
    if not can_prepare_awq(pack, "dense"):
        pytest.skip("Standalone Marlin modules for Hopper required")
    shifts = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device="cuda", dtype=torch.int32) * 4
    values = ((packed[..., None] >> shifts) & 15).view(height // 128, 128, width)
    decoded_zeros = ((zeros[..., None] >> shifts) & 15).view(height // 128, width)
    reference_weight = ((values - decoded_zeros[:, None]).float() * scales.float()[:, None]).view(height, width)
    activations = torch.randn(tokens, height, device="cuda", dtype=dtype)
    expected = activations.float() @ reference_weight
    original_ptr = packed.data_ptr()
    prepare_awq(pack)
    assert pack.weight.data_ptr() == original_ptr
    out = torch.empty(tokens, width, dtype=dtype, device="cuda")
    assert dense_awq_marlin(activations, pack, out) is out
    relative_error = (out.float() - expected).norm() / expected.norm()
    assert relative_error < 0.008


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "tokens,experts",
    [
        (1, 8),
        (16, 8),
        (8, 32),
        (9, 32),
        (48, 8),
        (48, 32),
        (64, 32),
        (95, 8),
        (96, 8),
        (96, 32),
        (112, 32),
        (113, 32),
        (144, 32),
        (145, 32),
        (159, 32),
        (160, 32),
        (192, 32),
        (193, 32),
        (128, 8),
        (128, 32),
        (512, 8),
        (513, 8),
    ],
)
def test_moe_marlin_awq_reduction(dtype, tokens, experts):
    torch.manual_seed(12)
    hidden, intermediate = 2048, 768
    shifts = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device="cuda", dtype=torch.int32) * 4

    def make_weights(height, width):
        groups = height // 128
        packed = torch.randint(-(2 ** 31), 2 ** 31 - 1, (experts, height, width // 8), dtype=torch.int32, device="cuda")
        zeros = torch.randint_like(packed[:, :groups], -(2 ** 31), 2 ** 31 - 1)
        scales = torch.rand(experts, groups, width, dtype=dtype, device="cuda") / 64
        pack = WeightPack(packed, scales, zeros)
        if not can_prepare_awq(pack, "moe"):
            pytest.skip("Standalone Marlin modules for Hopper required")
        values = ((packed[..., None] >> shifts) & 15).view(experts, groups, 128, width)
        decoded_zeros = ((zeros[..., None] >> shifts) & 15).view(experts, groups, 1, width)
        dense = ((values - decoded_zeros).float() * scales.float().unsqueeze(2)).view(experts, height, width)
        prepare_awq(pack)
        return pack, dense

    gate, dense_gate = make_weights(hidden, 2 * intermediate)
    down, dense_down = make_weights(intermediate, hidden)
    inputs = torch.randn(tokens, hidden, device="cuda", dtype=dtype)
    logits = torch.randn(tokens, experts, device="cuda")
    logits[:, 0] += 6
    scores, ids = logits.softmax(dim=-1).topk(2)
    ids = ids.to(torch.int32)

    def reference():
        expected = torch.empty(tokens, 2, hidden, device="cuda", dtype=dtype)
        for expert in range(experts):
            rows, slots = (ids == expert).nonzero(as_tuple=True)
            projected = (inputs[rows].float() @ dense_gate[expert]).to(dtype)
            gated, values = projected.chunk(2, dim=-1)
            activated = (torch.nn.functional.silu(gated.float()) * values.float()).to(dtype)
            expected[rows, slots] = ((activated.float() @ dense_down[expert]) * scores[rows, slots, None]).to(dtype)
        return expected.float().sum(dim=1).to(dtype).float()

    expected = reference()
    actual = moe_awq_marlin(inputs.clone(), gate, down, scores, ids, None, None, False)
    assert (actual.float() - expected).norm() / expected.norm() < 0.012

    if experts == 32:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = moe_awq_marlin(inputs.clone(), gate, down, scores, ids, None, None, False)
        for _ in range(100):
            graph.replay()
            assert torch.equal(replayed, actual)

    if tokens in (1, 8, 9, 16, 48, 64, 96, 112, 113, 144, 145, 159, 160, 192, 193, 512, 513):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = moe_awq_marlin(inputs.clone(), gate, down, scores, ids, None, None, False)
        for replay in range(24):
            inputs.normal_()
            logits.normal_()
            logits[:, replay % experts] += 6
            next_scores, next_ids = logits.softmax(dim=-1).topk(2)
            scores.copy_(next_scores)
            ids.copy_(next_ids)
            graph.replay()
            expected = reference()
            tolerance = 0.002 if dtype == torch.float16 else 0.01
            assert (replayed.float() - expected).norm() / expected.norm() < tolerance
