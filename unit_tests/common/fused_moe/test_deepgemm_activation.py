import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_fused_moe_ep as ep
from lightllm.common.basemodel.triton_kernel.fused_moe import moe_silu_and_mul
from lightllm.common.basemodel.triton_kernel.quantization import fp8act_quant_kernel


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not ep.HAS_DEEPGEMM or torch.cuda.get_device_capability()[0] < 9,
    reason="SM90+, DeepEP and DeepGEMM required",
)


def test_prefill_gather_initializes_and_accumulates_chunks():
    # Include a hidden tail, an unrouted token and a token with routes in both chunks.
    values = torch.arange(4 * 1031, device="cuda", dtype=torch.float32).reshape(4, 1031).to(torch.bfloat16)
    weights = torch.tensor([0.25, 0.5, 0.125, 0.75], device="cuda")
    metadata = torch.tensor([[0, 0, 0, 3], [0, 0, 2, -1], [0, 0, -1, -1]], device="cuda", dtype=torch.int32)
    out = torch.full_like(values[:3], torch.nan)
    ep.ep_gather_chunk(values[:2], 0, weights, metadata, out, initialize=True)
    first = (values[0].float() * weights[0]).bfloat16()
    expected = torch.zeros_like(out)
    expected[0] = first
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    ep.ep_gather_chunk(values[2:], 2, weights, metadata, out)
    expected[0] = first.float() + values[3].float() * weights[3]
    expected[1] = values[2].float() * weights[2]
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("use_sgl", [False, True])
def test_b128_prefill_fused_activation_matches_quantizer_bytes(monkeypatch, use_sgl):
    if use_sgl and not fp8act_quant_kernel.HAS_SGL_KERNEL:
        pytest.skip("sgl_kernel not installed")
    monkeypatch.setattr(fp8act_quant_kernel, "HAS_SGL_KERNEL", use_sgl)
    monkeypatch.setattr(moe_silu_and_mul, "HAS_SGL_KERNEL", use_sgl)
    torch.manual_seed(191)
    gate_up = torch.randn((128, 4608), dtype=torch.bfloat16, device="cuda") * 4
    gate_up[0].zero_()
    activated = torch.empty((128, 2304), dtype=gate_up.dtype, device="cuda")
    ep.silu_and_mul_fwd(gate_up, activated, limit=10.0)
    expected_q, expected_s = ep.per_token_group_quant_fp8(
        activated, 128, column_major_scales=True, scale_tma_aligned=True, use_ue8m0_scales=False
    )
    q, s = ep.silu_and_mul_quant_fp8(gate_up, 128, torch.float8_e4m3fn, limit=10.0, use_ue8m0_scales=False)
    torch.testing.assert_close(q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(s, expected_s, rtol=0, atol=0)


@pytest.mark.parametrize("is_prefill", [False, True])
@pytest.mark.parametrize("alpha,limit,add_one", [(None, None, True), (1.0, 10.0, False), (1.702, 7.0, True)])
def test_grouped_gemm_activation_matches_reference(monkeypatch, is_prefill, alpha, limit, add_one):
    # Identity projections isolate activation/FP8 behavior from GEMM rounding.
    dim, padded = 128, 128
    x = torch.linspace(-24, 24, 5 * dim, device="cuda").view(5, dim).to(torch.float8_e4m3fn)
    eye = torch.eye(dim, device="cuda", dtype=torch.bfloat16)
    w1 = torch.cat([2 * eye, 3 * eye]).unsqueeze(0).repeat(2, 1, 1).to(torch.float8_e4m3fn)
    w2 = eye.unsqueeze(0).repeat(2, 1, 1).to(torch.float8_e4m3fn)
    w1_scale = torch.ones(2, 2, 1, device="cuda")
    w2_scale = torch.ones(2, 1, 1, device="cuda")
    counts = torch.tensor([3, 2], device="cuda", dtype=torch.int32)
    activation_args = dict(alpha=alpha, limit=limit, clamp_up_add_one=add_one)
    recv = torch.zeros(2, padded, dim, device="cuda", dtype=torch.float8_e4m3fn)
    recv[0, :3] = x[:3]
    recv[1, :2] = x[3:]

    if is_prefill:
        metadata = torch.zeros(5, 3, device="cuda", dtype=torch.int32)
        metadata[:, 2] = torch.tensor([0, 1, 2, padded, padded + 1], device="cuda")
        # Force two chunks so both expert groups execute the requested activation.
        monkeypatch.setattr(ep, "_get_max_chunk_rows", lambda **kwargs: padded)
        actual = ep.chunked_expanded_moe_forward(
            num_recv_tokens_per_expert_list=[padded, padded],
            num_unaligned_recv_tokens_per_expert=counts,
            recv_x=(recv.view(2 * padded, dim), torch.ones(1, 2 * padded, device="cuda").T),
            recv_topk_weights=torch.ones(2 * padded, device="cuda"),
            recv_src_metadata=metadata,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            block_size_k=128,
            workspace=torch.empty(4 * 1024 * 1024, device="cuda", dtype=torch.uint8),
            hidden_dtype=torch.bfloat16,
            **activation_args,
        )
    else:
        output = ep.masked_group_gemm(
            (recv, torch.ones(2, padded, 1, device="cuda")),
            counts,
            torch.bfloat16,
            w1,
            w1_scale,
            w2,
            w2_scale,
            expected_m=3,
            **activation_args,
        )
        actual = torch.cat([output[0, :3], output[1, :2]])

    gate = (x.float() * 2).bfloat16().float()
    up = (x.float() * 3).bfloat16().float()
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit) + int(add_one)
    gate = (gate * torch.sigmoid(gate * (alpha or 1.0))).bfloat16().float()
    activation = (gate * up).bfloat16().float()
    scales = activation.abs().amax(-1, keepdim=True).clamp(min=1e-10) / 448
    expected = ((activation / scales).to(torch.float8_e4m3fn).float() * scales).bfloat16()
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.02)


def test_group32_prefill_activation_across_workspace_chunks():
    dim, alignment = 128, 128
    counts = [3, 0, 2]
    active = [0, 1, 2, alignment, alignment + 1]
    fp8 = torch.float8_e4m3fn
    torch.manual_seed(154)
    x = (torch.randn((len(active), dim), device="cuda") * 5).to(fp8)
    recv = torch.zeros((2 * alignment, dim), dtype=fp8, device="cuda")
    recv.view(torch.uint8)[active] = x.view(torch.uint8)
    eye = torch.eye(dim, device="cuda")
    w1 = torch.cat([eye, 2 * eye]).repeat(len(counts), 1, 1).to(fp8)
    w2 = eye.repeat(len(counts), 1, 1).to(fp8)
    unit_scales = torch.full((len(counts), 2 * dim, dim // 32), 127, dtype=torch.uint8, device="cuda")
    metadata = torch.zeros((len(active), 3), device="cuda", dtype=torch.int32)
    metadata[:, 2] = torch.tensor(active, device="cuda")
    # The cached gather reservation rounds up to 1024 rows. The remainder fits exactly one 128-row chunk.
    workspace_bytes = 1024 * dim * 2 + alignment * (4 * dim + dim + 4 * (dim // 32))
    workspace = torch.empty(workspace_bytes, device="cuda", dtype=torch.uint8)
    actual = ep.chunked_expanded_moe_forward(
        [alignment, 0, alignment],
        torch.tensor(counts, dtype=torch.int32, device="cuda"),
        (recv, torch.ones((dim // 32, len(recv)), device="cuda").T),
        torch.ones(len(recv), device="cuda"),
        metadata,
        w1,
        unit_scales,
        w2,
        unit_scales[:, :dim].contiguous(),
        32,
        workspace,
        torch.bfloat16,
        limit=10.0,
    )
    gate_up = torch.cat((x.float(), x.float() * 2), dim=1).bfloat16()
    activation = torch.empty_like(gate_up[:, :dim], memory_format=torch.contiguous_format)
    ep.silu_and_mul_fwd(gate_up, activation, limit=10.0)
    q, s = ep.per_token_group_quant_fp8(activation, 32, use_ue8m0_scales=True)
    expected = (q.float() * s.repeat_interleave(32, 1)).bfloat16()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert metadata[:, 2].tolist() == list(range(len(active)))
