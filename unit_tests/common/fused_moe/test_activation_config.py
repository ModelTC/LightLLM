import dataclasses
import json

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight
from lightllm.common.quantization.no_quant import NoQuantization
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import get_env_start_args


@pytest.fixture(autouse=True)
def runtime(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(dataclasses.asdict(StartArgs())))
    for name, value in {
        "GLOBAL_RANK": 0,
        "GLOBAL_WORLD_SIZE": 1,
        "DP_WORLD_SIZE": 1,
        "CURRENT_RANK_IN_DP": 0,
        "CURRENT_RANK_IN_NODE": 0,
        "CURRENT_DEVICE_ID": 0,
    }.items():
        monkeypatch.setenv("LIGHTLLM_" + name, str(value))
    get_env_start_args.cache_clear()
    yield
    get_env_start_args.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("activation", ["silu", "clamped_silu", "clamped_silu_add_one", "gelu"])
def test_call_parameters_reach_expert_activation(monkeypatch, activation):
    monkeypatch.setattr(
        "lightllm.common.basemodel.triton_kernel.fused_moe.moe_silu_and_mul.ffn_use_tanh_approximate_gelu",
        lambda: activation == "gelu",
    )
    dim, count = 128, 4
    config = {"norm_topk_prob": True, "num_experts_per_tok": 2, "scoring_func": "softmax"}
    kwargs = {
        "clamped_silu": {"alpha": 1.0, "limit": 10.0, "clamp_up_add_one": False},
        "clamped_silu_add_one": {"alpha": 1.702, "limit": 7.0},
    }.get(activation, {})
    weight = FusedMoeWeight(
        gate_proj_name="gate",
        up_proj_name="up",
        down_proj_name="down",
        e_score_correction_bias_name="",
        weight_prefix="experts",
        n_routed_experts=count,
        hidden_size=dim,
        moe_intermediate_size=dim,
        data_type=torch.bfloat16,
        quant_method=NoQuantization(),
        network_config=config,
    )
    eye = torch.eye(dim, device="cuda", dtype=torch.bfloat16)
    weights = {}
    for i in range(count):
        weights[f"experts.{i}.gate.weight"] = eye * (1 + i / 4)
        weights[f"experts.{i}.up.weight"] = eye * (2 + i / 8)
        weights[f"experts.{i}.down.weight"] = eye
    weight.load_hf_weights(weights)
    assert weight.verify_load()
    x = torch.linspace(-25, 25, 3 * dim, device="cuda", dtype=torch.bfloat16).view(3, dim)
    router = torch.tensor([[1, 3, 2, 0], [4, 3, 1, 2], [1, 2, 3, 4]], device="cuda", dtype=torch.float32)
    top = router.topk(2, dim=-1)
    probs = top.values.softmax(-1)
    expected = torch.zeros_like(x, dtype=torch.float32)
    for row in range(x.shape[0]):
        for choice in range(2):
            i = int(top.indices[row, choice])
            gate = (x[row] * (1 + i / 4)).float()
            up = (x[row] * (2 + i / 8)).float()
            if kwargs:
                gate = gate.clamp(max=kwargs["limit"])
                up = up.clamp(-kwargs["limit"], kwargs["limit"])
                gate = gate * torch.sigmoid(kwargs["alpha"] * gate)
                if kwargs.get("clamp_up_add_one", True):
                    up += 1
            elif activation == "gelu":
                gate = torch.nn.functional.gelu(gate, approximate="tanh")
            else:
                gate = torch.nn.functional.silu(gate)
            expert_out = (gate.bfloat16() * up.bfloat16()).bfloat16()
            expected[row] += (expert_out.float() * probs[row, choice]).bfloat16().float()
    if kwargs:
        default_output = weight.experts(x.clone(), router, 2, True, False, 0, 0, False)
    actual = weight.experts(x.clone(), router, 2, True, False, 0, 0, False, **kwargs)
    actual_with_topk = weight.experts_with_topk(
        x.clone(),
        topk_weights=probs,
        topk_ids=top.indices,
        is_prefill=False,
        **kwargs,
    )
    torch.testing.assert_close(actual, expected.bfloat16(), atol=0.125, rtol=0.01)
    torch.testing.assert_close(actual_with_topk, expected.bfloat16(), atol=0.125, rtol=0.01)
    if kwargs:
        # A clamped call must not change subsequent calls on the same weight.
        actual_default = weight.experts(x.clone(), router, 2, True, False, 0, 0, False)
        torch.testing.assert_close(actual_default, default_output, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packed_awq_dense_and_experts(dtype):
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe import fused_experts_impl
    from lightllm.common.basemodel.triton_kernel.quantization.awq_gemm import awq_gemm

    torch.manual_seed(100)
    group_size = 32
    permutation = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device="cuda")
    shifts = torch.arange(0, 32, 4, device="cuda")

    def pack(values):
        return (values.unflatten(-1, (-1, 8))[..., permutation] << shifts).sum(-1).int()

    def make_weights(experts, input_width, output_width):
        weight = torch.randint(0, 16, (experts, input_width, output_width), device="cuda")
        zeros = torch.randint(0, 16, (experts, input_width // group_size, output_width), device="cuda")
        scales = torch.rand(zeros.shape, device="cuda", dtype=dtype) / 64
        reference = (
            (weight - zeros.repeat_interleave(group_size, dim=1)) * scales.repeat_interleave(group_size, dim=1)
        ).to(dtype)
        return pack(weight), scales, pack(zeros), reference

    inputs = torch.randn(7, 64, device="cuda", dtype=dtype)
    w1, s1, z1, reference1 = make_weights(3, 64, 128)
    w2, s2, z2, reference2 = make_weights(3, 64, 64)
    dense = awq_gemm(inputs, w1[0, :, 3:11], s1[0, :, 24:88], z1[0, :, 3:11])
    torch.testing.assert_close(dense, inputs @ reference1[0, :, 24:88], atol=0.002, rtol=0.015)

    expert_ids = torch.tensor(
        [[0, 2], [1, 0], [2, 1], [0, 1], [1, 2], [2, 0], [1, 0]], device="cuda", dtype=torch.int32
    )
    router_weights = torch.rand(expert_ids.shape, device="cuda", dtype=torch.float32)
    actual = fused_experts_impl(
        inputs, w1, w2, router_weights, expert_ids, w1_scale=s1, w2_scale=s2, w1_zeros=z1, w2_zeros=z2
    )
    gate, up = torch.einsum("tk,tekn->ten", inputs, reference1[expert_ids.long()]).chunk(2, dim=-1)
    hidden = (torch.nn.functional.silu(gate.float()) * up.float()).to(dtype)
    expert_values = torch.einsum("tek,tekn->ten", hidden, reference2[expert_ids.long()])
    expected = (expert_values.float() * router_weights[..., None]).sum(dim=1).to(dtype)
    torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.035)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "name,canonical,dtype",
    [("w8a8-vllm", "w8a8", torch.int8), ("fp8w8a8-vllm", "fp8w8a8", torch.float8_e4m3fn)],
)
def test_per_channel_quantization_legacy_alias(name, canonical, dtype):
    from lightllm.common.quantization.quantize_method import WeightPack
    from lightllm.common.quantization.registry import QUANTMETHODS

    torch.manual_seed(34)
    method = QUANTMETHODS.get(name)
    assert type(method) is type(QUANTMETHODS.get(canonical))
    inputs = torch.randn(19, 128, device="cuda", dtype=torch.bfloat16)
    inputs[0].zero_()
    weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16) / 16
    bias = torch.randn(128, device="cuda", dtype=torch.bfloat16) / 16
    weights = WeightPack(weight=torch.empty_like(weight, dtype=dtype), weight_scale=torch.empty(128, device="cuda"))
    method.quantize(weight, weights)
    out = torch.empty_like(inputs)
    actual = method.apply(inputs, weights, out=out, bias=bias)
    assert actual is out
    limit = 127 if dtype == torch.int8 else torch.finfo(dtype).max
    values = inputs.float()
    scale = (values.abs().amax(dim=-1, keepdim=True) / limit).clamp_min(1e-13)
    normalized = (values / scale).clamp(-limit, limit)
    if dtype == torch.int8:
        normalized = normalized.round()
    dequantized = normalized.to(dtype).float() * scale
    reference_weights = weights.weight.float() * weights.weight_scale[:, None]
    expected = (dequantized @ reference_weights.T).bfloat16() + bias
    torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("static_scale", [None, 0.025])
def test_grouped_fp8_per_token_scales(static_scale):
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe import grouped_matmul

    torch.manual_seed(8)
    inputs = torch.randn(19, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(2, 128, 128, device="cuda").to(torch.float8_e4m3fn)
    weight_scale = torch.rand(2, 128, device="cuda") / 64
    expert_counts = torch.tensor([19, 0], device="cuda", dtype=torch.int32)
    token_indices = torch.arange(19, device="cuda", dtype=torch.int32).repeat(2, 1)
    route_weights = torch.ones_like(token_indices, dtype=torch.float32)
    scale = torch.tensor([static_scale], device="cuda") if static_scale is not None else None
    out = torch.empty_like(inputs)
    grouped_matmul(
        19,
        inputs,
        scale,
        expert_counts,
        token_indices,
        route_weights,
        weights,
        weight_scale,
        topk_num=1,
        out=out,
        mul_routed_weight=False,
        use_fp8_w8a8=True,
    )
    reference_scale = scale if scale is not None else inputs.float().abs().amax(dim=-1, keepdim=True) / 448
    dequantized = (inputs.float() / reference_scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * reference_scale
    reference_weight = weights[0].float() * weight_scale[0, :, None]
    expected = (dequantized @ reference_weight.T).to(inputs.dtype)
    torch.testing.assert_close(out, expected, atol=0.005, rtol=0.025)
