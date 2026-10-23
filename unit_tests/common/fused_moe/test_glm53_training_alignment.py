"""GLM launch recipe: eager BF16 boundaries, FP32 SwiGLU and p before W2."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul_mix_quant_ep import (
    silu_and_mul_masked_post_quant_fwd,
)
from lightllm.common.basemodel.triton_kernel.fused_moe.moe_route_weight import (
    multiply_routing_weights,
    gather_low_latency_routing_weights,
    _gather_low_latency_weights,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def swiglu_reference(x):
    gate, up = x.float().chunk(2, -1)
    return (F.silu(gate.clamp(max=10)) * up.clamp(-10, 10)).to(x.dtype)


def quant_reference(x, ue8m0):
    groups = x.float().reshape(*x.shape[:-1], -1, 128)
    sf = groups.abs().amax(-1).clamp_min(1e-10) / 448
    if ue8m0:
        sf = torch.exp2(torch.ceil(torch.log2(sf)))
    q = (groups / sf.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape_as(x), sf


@pytest.mark.parametrize("rows,width", [(1, 256), (19, 4096)])
def test_swiglu_retains_fp32_product_and_bf16_weight_boundary(rows, width):
    torch.manual_seed(53)
    x = (torch.randn(rows, width, device="cuda") * 6).bfloat16()
    out = torch.empty(rows, width // 2, device=x.device, dtype=x.dtype)
    silu_and_mul_fwd(x, out, alpha=1.0, limit=10.0, clamp_up_add_one=False, glm5_numerics=True)
    expected = swiglu_reference(x)
    # CUDA exp implementations differ near BF16 midpoints.
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0.008)
    old = torch.empty_like(out)
    silu_and_mul_fwd(x, old, alpha=1.0, limit=10.0, clamp_up_add_one=False)
    assert torch.count_nonzero(old != expected) > torch.count_nonzero(out != expected)
    p = torch.linspace(0.01, 1.7, rows, device=x.device)
    before = out.clone()
    multiply_routing_weights(out, p)
    torch.testing.assert_close(out, (before.float() * p[:, None]).to(x.dtype), atol=0, rtol=0)


@pytest.mark.parametrize("ue8m0", [False, True])
def test_masked_decode_swiglu_matches_training_before_quant(monkeypatch, ue8m0):
    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", str(int(ue8m0)))
    torch.manual_seed(53)
    x = (torch.randn(3, 5, 4096, device="cuda") * 6).bfloat16()
    p = torch.rand(3, 5, device=x.device) * 1.7
    p[1, 0] = 0
    counts = torch.tensor([0, 2, 5], device=x.device, dtype=torch.int32)
    out = torch.full((3, 5, 2048), 1, device=x.device, dtype=torch.float8_e4m3fn)
    sf = torch.full((3, 5, 16), 7.0, device=x.device)
    silu_and_mul_masked_post_quant_fwd(
        x,
        out,
        sf,
        128,
        counts,
        alpha=1.0,
        limit=10.0,
        clamp_up_add_one=False,
        glm5_numerics=True,
        routing_weights=p,
    )
    weighted = (swiglu_reference(x).float() * p[..., None]).to(x.dtype)
    q_ref, s_ref = quant_reference(weighted, ue8m0)
    for expert, count in enumerate(counts.tolist()):
        torch.testing.assert_close(sf[expert, :count], s_ref[expert, :count], atol=1e-6, rtol=1e-5)
        actual = out[expert, :count].float() * sf[expert, :count].repeat_interleave(128, -1)
        expected = q_ref[expert, :count].float() * s_ref[expert, :count].repeat_interleave(128, -1)
        # A SwiGLU midpoint may flip one FP8 rounding decision.
        err = (actual - expected).square().mean().sqrt() if count else actual.sum()
        assert err < 0.01
        assert torch.all(out[expert, count:].float() == 1)
        assert torch.all(sf[expert, count:] == 7)


@pytest.mark.parametrize("empty_local", [False, True])
def test_low_latency_routing_uses_source_and_actual_receive_order(monkeypatch, empty_local):
    from lightllm.common.basemodel.triton_kernel.fused_moe import moe_route_weight as route

    # Two ranks, unequal token counts; expert receive order differs by source rank.
    max_tokens, ranks, local_experts, topk = 4, 2, 2, 2
    ids = torch.tensor([[[2, 3], [0, 2], [-1, -1], [-1, -1]], [[3, 2], [2, 1], [2, 3], [-1, -1]]], device="cuda")
    p = torch.tensor(
        [[[0.1, 0.9], [0.2, 0.8], [0, 0], [0, 0]], [[0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0, 0]]], device="cuda"
    )
    if empty_local:
        ids[1].fill_(-1)
        p[1].zero_()
    routes = torch.stack((ids.float(), p), -1).reshape(ranks * max_tokens, topk, 2)
    src_info = torch.zeros((local_experts, ranks * max_tokens), device="cuda", dtype=torch.int32)
    layout = torch.zeros((local_experts, ranks), device="cuda", dtype=torch.int64)
    expected = torch.zeros_like(src_info, dtype=torch.float32)
    rank = 1
    for expert in range(local_experts):
        offset = 0
        for src_rank in (1, 0):
            tokens = torch.where((ids[src_rank] == rank * local_experts + expert).any(-1))[0].flip(0)
            count = tokens.numel()
            src_info[expert, offset : offset + count] = tokens.int()
            layout[expert, src_rank] = (offset << 32) | count
            for row, token in enumerate(tokens.tolist()):
                slot = (ids[src_rank, token] == rank * local_experts + expert).nonzero()[0, 0]
                expected[expert, offset + row] = p[src_rank, token, slot]
            offset += count
    calls = []

    def gather(out, inp, group):
        calls.append(inp.clone())
        out.copy_(routes)

    monkeypatch.setattr(route.dist, "all_gather_into_tensor", gather)
    buffer = SimpleNamespace(group_size=ranks, rank=rank, group=object())
    handle = (src_info, layout, max_tokens, 256, ranks * local_experts)
    actual = gather_low_latency_routing_weights(
        buffer, ids[1, : 0 if empty_local else 3], p[1, : 0 if empty_local else 3], handle
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(calls) == 1
    assert torch.all(calls[0][3, :, 0] == -1)
    # The metadata mapping is graph replayable and reads changed inputs.
    out = torch.zeros_like(expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            _gather_low_latency_weights[(local_experts, ranks)](
                routes, src_info, layout, out, ranks, max_tokens, topk, local_experts, rank, 32, 2
            )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _gather_low_latency_weights[(local_experts, ranks)](
            routes, src_info, layout, out, ranks, max_tokens, topk, local_experts, rank, 32, 2
        )
    routes[:, :, 1].mul_(2)
    graph.replay()
    torch.testing.assert_close(out, expected * 2, atol=0, rtol=0)


@pytest.mark.parametrize("tokens", [2, 19, 257])
@pytest.mark.parametrize("tma", [False, True])
@pytest.mark.parametrize(
    "quantized,ue8m0,packed", [(False, False, False), (True, False, False), (True, True, False), (True, True, True)]
)
def test_tp_moe_routes_before_w2_with_tma_sorting_and_shared_expert(
    monkeypatch, tma, quantized, ue8m0, packed, tokens
):
    from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_fused_moe as tp

    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", str(int(ue8m0 and not packed)))
    monkeypatch.setattr(tp, "triton_support_tensor_descriptor", lambda: tma)
    torch.manual_seed(53)
    dim, experts = 128, 4
    x = (torch.randn(tokens, dim, device="cuda") * 4).bfloat16()
    ids = torch.stack(
        (
            torch.arange(tokens, device=x.device) % 3,
            (torch.arange(tokens, device=x.device) + 1) % 3,
            torch.full((tokens,), 3, device=x.device),
        ),
        -1,
    ).int()
    p = torch.rand(tokens, 3, device=x.device) * 1.7
    p[:, -1] = 1  # Fused shared expert must remain unweighted.
    eye = torch.eye(dim, device=x.device, dtype=x.dtype)
    w1 = torch.cat((eye, eye)).repeat(experts, 1, 1)
    w2 = eye.repeat(experts, 1, 1)
    kwargs = {}

    def quantize(x):
        # Ordinary scales use the installed SGL quantizer; its reciprocal
        # multiplication can resolve FP8 midpoints differently from Torch's
        # division. Keep that unchanged stage fixed while testing W2 placement.
        if ue8m0:
            return quant_reference(x, True)
        from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import per_token_group_quant_fp8

        q, sf = per_token_group_quant_fp8(x.reshape(-1, dim), 128)
        return q.reshape_as(x), sf.reshape(*x.shape[:-1], 1)

    if quantized:
        w1, w2 = w1.to(torch.float8_e4m3fn), w2.to(torch.float8_e4m3fn)
        kwargs = dict(
            w1_scale=torch.ones(experts, 2, 1, device=x.device), w2_scale=torch.ones(experts, 1, 1, device=x.device)
        )
        if packed:
            # One K group: exponent 127 occupies byte 0, per output channel.
            # Leave the env off to check packed B itself implies UE8M0 on A.
            kwargs = dict(
                w1_scale=torch.full((experts, 256, 1), 127, device=x.device, dtype=torch.int32),
                w2_scale=torch.full((experts, 128, 1), 127, device=x.device, dtype=torch.int32),
            )
        q, sf = quantize(x)
        pre = (q.float() * sf.repeat_interleave(128, -1)).bfloat16()
    else:
        pre = x
    h = swiglu_reference(torch.cat((pre, pre), -1))
    h = (h[:, None, :].float() * p[..., None]).bfloat16()
    if quantized:
        q, sf = quantize(h)
        h = (q.float() * sf.repeat_interleave(128, -1)).bfloat16()
    expected = h.float().sum(1).bfloat16()
    actual = tp.fused_experts(
        x.clone(),
        w1,
        w2,
        p,
        ids,
        use_fp8_w8a8=quantized,
        alpha=1.0,
        limit=10.0,
        clamp_up_add_one=False,
        glm5_numerics=True,
        **kwargs,
    )
    torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.008)
    legacy = tp.fused_experts(
        x.clone(),
        w1,
        w2,
        p,
        ids,
        use_fp8_w8a8=quantized,
        alpha=1.0,
        limit=10.0,
        clamp_up_add_one=False,
        **kwargs,
    )
    assert torch.count_nonzero(legacy != expected) > torch.count_nonzero(actual != expected)


@pytest.mark.parametrize("ue8m0", [False, True])
def test_ep_decode_dispatch_to_combine_weights_exactly_once(monkeypatch, ue8m0):
    from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_fused_moe_ep as ep

    monkeypatch.setenv("LIGHTLLM_USE_UE8M0_SCALES", str(int(ue8m0)))
    monkeypatch.setattr(ep, "get_deepep_num_max_dispatch_tokens_per_rank_decode", lambda: 128)
    torch.manual_seed(53)
    tokens, dim, experts = 5, 128, 2
    x = (torch.randn(tokens, dim, device="cuda") * 3).bfloat16()
    ids = torch.tensor([[0, 1]] * tokens, device=x.device, dtype=torch.int64)
    p = torch.rand(tokens, 2, device=x.device) * 1.7
    orders = torch.tensor([[4, 1, 3, 0, 2], [2, 4, 1, 3, 0]], device=x.device)
    eye = torch.eye(dim, device=x.device, dtype=x.dtype)
    w1 = torch.cat((2 * eye, 3 * eye)).repeat(experts, 1, 1).to(torch.float8_e4m3fn)
    w2 = eye.repeat(experts, 1, 1).to(torch.float8_e4m3fn)
    seen = []

    class Buffer:
        group_size, rank = 1, 0

        def low_latency_dispatch(self, hidden, topk_idx, max_tokens, num_experts, **kwargs):
            assert kwargs["round_scale"] == ue8m0
            q, sf = quant_reference(hidden, ue8m0)
            recv = torch.zeros(experts, max_tokens, dim, device=x.device, dtype=q.dtype)
            scales = torch.zeros(experts, max_tokens, 1, device=x.device)
            src = torch.zeros(experts, max_tokens, device=x.device, dtype=torch.int32)
            for e in range(experts):
                recv[e, :tokens] = q[orders[e]]
                scales[e, :tokens] = sf[orders[e]]
                src[e, :tokens] = orders[e].int()
            counts = torch.full((experts,), tokens, device=x.device, dtype=torch.int32)
            layout = torch.full((experts, 1), tokens, device=x.device, dtype=torch.int64)
            return (recv, scales), counts, (src, layout, max_tokens, dim, experts), None, None

        def low_latency_combine(self, output, topk_idx, topk_weights, handle, **kwargs):
            torch.testing.assert_close(topk_weights, torch.ones_like(p), atol=0, rtol=0)
            seen.append(True)
            result = torch.zeros(tokens, dim, device=x.device)
            for e in range(experts):
                result[orders[e]] += output[e, :tokens].float()
            return result.bfloat16(), None, None

    actual = ep.fused_experts_impl(
        x,
        w1,
        w2,
        p,
        ids,
        experts,
        Buffer(),
        False,
        use_fp8_w8a8=True,
        use_fp8_all2all=True,
        w1_scale=torch.ones(experts, 2, 1, device=x.device),
        w2_scale=torch.ones(experts, 1, 1, device=x.device),
        alpha=1.0,
        limit=10.0,
        clamp_up_add_one=False,
        glm5_numerics=True,
    )
    q, sf = quant_reference(x, ue8m0)
    pre = (q.float() * sf.repeat_interleave(128, -1)).bfloat16()
    h = swiglu_reference(torch.cat(((2 * pre.float()).bfloat16(), (3 * pre.float()).bfloat16()), -1))
    weighted = (h[:, None, :].float() * p[..., None]).bfloat16()
    q, sf = quant_reference(weighted, ue8m0)
    expected = (q.float() * sf.repeat_interleave(128, -1)).bfloat16().float().sum(1).bfloat16()
    if ue8m0:
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.008)
    else:
        # Arbitrary scales use reciprocal/division implementations with distinct
        # FP8 midpoint ties. Bound the difference by one FP8 step per route,
        # plus the final BF16 sum rounding; UE8M0 keeps the strict comparison.
        q_float = q.float().abs().clamp_min(2.0**-6)
        fp8_step = torch.exp2(torch.floor(torch.log2(q_float)) - 3) * sf.repeat_interleave(128, -1)
        bound = fp8_step.sum(1) + expected.float().abs() * torch.finfo(torch.bfloat16).eps
        assert torch.all((actual.float() - expected.float()).abs() <= bound + 1e-5)
        nrmse = (actual.float() - expected.float()).square().mean().sqrt() / expected.float().square().mean().sqrt()
        assert nrmse < 0.008
    assert seen == [True]
