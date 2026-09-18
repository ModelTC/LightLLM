import dataclasses
import json

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.deepgemm_impl import FuseMoeDeepGEMM
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.marlin_impl import FuseMoeMarlin
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
        default_output = weight.experts(x.clone(), router, 2, True, False, 0, 0)
    actual = weight.experts(x.clone(), router, 2, True, False, 0, 0, **kwargs)
    torch.testing.assert_close(actual, expected.bfloat16(), atol=0.125, rtol=0.01)
    if kwargs:
        # A clamped call must not change subsequent calls on the same weight.
        actual_default = weight.experts(x.clone(), router, 2, True, False, 0, 0)
        torch.testing.assert_close(actual_default, default_output, atol=0, rtol=0)


@pytest.mark.parametrize("backend", [FuseMoeDeepGEMM, FuseMoeMarlin])
def test_unsupported_backend_rejects_clamp_at_call(monkeypatch, backend):
    monkeypatch.setattr(backend, "create_workspace", lambda self: None)
    monkeypatch.setattr(backend, "_select_experts", lambda *args, **kwargs: (None, None, None))
    impl = backend(
        n_routed_experts=4,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
        quant_method=None,
        redundancy_expert_num=0,
        redundancy_expert_ids_tensor=None,
        routed_expert_counter_tensor=None,
        auto_update_redundancy_expert=False,
    )
    with pytest.raises(NotImplementedError, match="does not support clamped SwiGLU"):
        impl(
            input_tensor=None,
            router_logits=None,
            w13=None,
            w2=None,
            correction_bias=None,
            scoring_func="softmax",
            top_k=2,
            renormalize=True,
            use_grouped_topk=False,
            topk_group=0,
            num_expert_group=0,
            alpha=1.0,
            limit=10.0,
            clamp_up_add_one=False,
        )
