import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_fused_moe_ep as ep


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not ep.HAS_DEEPGEMM or torch.cuda.get_device_capability()[0] < 9,
    reason="SM90+, DeepEP and DeepGEMM required",
)


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
