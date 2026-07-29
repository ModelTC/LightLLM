import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _unpack_mxfp4(packed, scale):
    code = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)
    magnitude_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=packed.device,
    )
    magnitude = magnitude_lut[(code & 0x7).long()]
    value = torch.where((code & 0x8).bool(), -magnitude, magnitude)
    scale = torch.exp2(scale.float() - 127).repeat_interleave(32, dim=-1)
    return (value * scale).to(torch.bfloat16)


def test_mxfp4_moe_matches_bf16_reference():
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_mxfp4 import (
        fused_experts_mxfp4,
    )

    torch.manual_seed(7)
    device = torch.device("cuda")
    experts, hidden, intermediate = 4, 64, 32
    token_count, topk = 3, 2

    w13 = torch.randint(0, 256, (experts, 2 * intermediate, hidden // 2), dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (experts, hidden, intermediate // 2), dtype=torch.uint8, device=device)
    w13_scale = torch.randint(120, 126, (experts, 2 * intermediate, hidden // 32), dtype=torch.uint8, device=device)
    w2_scale = torch.randint(120, 126, (experts, hidden, intermediate // 32), dtype=torch.uint8, device=device)
    x = 0.25 * torch.randn(token_count, hidden, dtype=torch.bfloat16, device=device)
    topk_ids = torch.tensor([[0, 2], [3, 1], [1, 0]], dtype=torch.int64, device=device)
    topk_weights = torch.rand(token_count, topk, dtype=torch.float32, device=device)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    dense_w13 = _unpack_mxfp4(w13, w13_scale)
    dense_w2 = _unpack_mxfp4(w2, w2_scale)
    expected = torch.zeros_like(x)
    for token_idx in range(token_count):
        for route_idx in range(topk):
            expert_idx = topk_ids[token_idx, route_idx]
            gate_up = torch.mv(dense_w13[expert_idx].float(), x[token_idx].float()).to(torch.bfloat16)
            gate, up = gate_up.float().chunk(2)
            activated = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate) * (25.0 * torch.tanh(up / 25.0))).to(
                torch.bfloat16
            )
            expert_out = torch.mv(dense_w2[expert_idx].float(), activated.float()).to(torch.bfloat16)
            expected[token_idx] += expert_out * topk_weights[token_idx, route_idx]

    actual = fused_experts_mxfp4(
        hidden_states=x.clone(),
        w1=w13,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        activation="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )

    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.1)
