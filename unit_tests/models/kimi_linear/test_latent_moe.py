from types import SimpleNamespace

import torch

from lightllm.models.kimi_linear.layer_infer.transformer_layer_infer import (
    KimiLinearTransformerLayerInfer,
)
from lightllm.models.kimi_linear.mem_manager import KimiLinearMemManager


class _Linear:
    def __init__(self, weight):
        self.weight = weight
        self.data_type_ = weight.dtype
        self.inputs = []

    def mm(self, input):
        self.inputs.append(input.clone())
        return input @ self.weight


class _Experts:
    def __init__(self, scale):
        self.scale = scale
        self.inputs = []

    def experts(self, input_tensor, **kwargs):
        self.inputs.append(input_tensor.clone())
        return input_tensor * self.scale


class _OutputProjection:
    def mm(self, input):
        return input


class _IdentityNorm:
    def __call__(self, input, **kwargs):
        return input


def test_latent_moe_routes_at_model_width_and_runs_experts_at_latent_width():
    infer = object.__new__(KimiLinearTransformerLayerInfer)
    infer.embed_dim_ = 4
    infer.n_shared_experts = None
    infer.num_experts_per_tok = 2
    infer.norm_topk_prob = True
    infer.n_group = 1
    infer.topk_group = 1
    infer.latent_moe_use_norm = False

    gate = _Linear(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [0.5, 0.5]]))
    down = _Linear(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.5]]))
    up = _Linear(torch.tensor([[1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 1.0, 2.0]]))
    experts = _Experts(scale=3.0)
    layer_weight = SimpleNamespace(
        moe_gate=gate,
        moe_latent_down_proj=down,
        moe_latent_up_proj=up,
        experts=experts,
    )
    hidden_states = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])

    actual = infer._latent_moe_ffn(hidden_states, infer_state=None, layer_weight=layer_weight)
    expected = (hidden_states @ down.weight * 3.0) @ up.weight

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(gate.inputs[0], hidden_states)
    assert experts.inputs[0].shape == (2, 2)


def test_latent_moe_normalizes_routed_output_before_up_projection():
    infer = object.__new__(KimiLinearTransformerLayerInfer)
    infer.embed_dim_ = 2
    infer.n_shared_experts = None
    infer.num_experts_per_tok = 1
    infer.norm_topk_prob = True
    infer.n_group = 1
    infer.topk_group = 1
    infer.latent_moe_use_norm = True
    infer.eps_ = 1e-6
    infer.alloc_tensor = torch.empty

    gate = _Linear(torch.ones((2, 1)))
    identity = _Linear(torch.eye(2))
    experts = _Experts(scale=2.0)

    class _Norm:
        def __init__(self):
            self.inputs = []

        def __call__(self, input, eps, alloc_func):
            self.inputs.append(input.clone())
            return input + 5.0

    norm = _Norm()
    layer_weight = SimpleNamespace(
        moe_gate=gate,
        moe_latent_down_proj=identity,
        moe_latent_norm=norm,
        moe_latent_up_proj=identity,
        experts=experts,
    )
    hidden_states = torch.tensor([[1.0, 3.0]])

    actual = infer._latent_moe_ffn(hidden_states, infer_state=None, layer_weight=layer_weight)

    torch.testing.assert_close(norm.inputs[0], hidden_states * 2.0)
    torch.testing.assert_close(actual, hidden_states * 2.0 + 5.0)


def test_latent_moe_reduces_tp_shards_before_normalization():
    infer = object.__new__(KimiLinearTransformerLayerInfer)
    infer.embed_dim_ = 2
    infer.n_shared_experts = None
    infer.num_experts_per_tok = 1
    infer.norm_topk_prob = True
    infer.n_group = 1
    infer.topk_group = 1
    infer.latent_moe_use_norm = True
    infer.eps_ = 1e-6
    infer.alloc_tensor = torch.empty
    infer._tpsp_reduce = lambda input, infer_state: input + 7.0

    gate = _Linear(torch.ones((2, 1)))
    identity = _Linear(torch.eye(2))
    experts = _Experts(scale=2.0)

    class _Norm:
        def __init__(self):
            self.inputs = []

        def __call__(self, input, eps, alloc_func):
            self.inputs.append(input.clone())
            return input

    norm = _Norm()
    layer_weight = SimpleNamespace(
        moe_gate=gate,
        moe_latent_down_proj=identity,
        moe_latent_norm=norm,
        moe_latent_up_proj=identity,
        experts=experts,
    )
    hidden_states = torch.tensor([[1.0, 3.0]])

    actual = infer._latent_moe_ffn(
        hidden_states,
        infer_state=None,
        layer_weight=layer_weight,
        reduce_tp_output=True,
    )

    expected = hidden_states * 2.0 + 7.0
    torch.testing.assert_close(norm.inputs[0], expected)
    torch.testing.assert_close(actual, expected)


def test_gated_mla_applies_sigmoid_before_output_projection():
    infer = object.__new__(KimiLinearTransformerLayerInfer)
    infer.use_gated_mla = True
    infer.kv_lora_rank = 3
    infer.tp_q_head_num_ = 2
    infer.v_head_dim = 2
    infer._tpsp_reduce = lambda input, infer_state: input

    gate = torch.tensor([[0.0, 1.0, -1.0, 2.0]])
    infer_state = SimpleNamespace(mla_output_gate=gate, need_dp_prefill_balance=False)
    layer_weight = SimpleNamespace(o_weight_=_OutputProjection())
    attention_output = torch.arange(1.0, 5.0).view(1, 2, 2)

    actual = infer._get_o(attention_output, infer_state, layer_weight)
    expected = torch.arange(1.0, 5.0).view(1, 4) * gate.sigmoid()

    torch.testing.assert_close(actual, expected)
    assert infer_state.mla_output_gate is None


def test_q_lora_path_builds_query_and_compressed_kv():
    infer = object.__new__(KimiLinearTransformerLayerInfer)
    infer.embed_dim_ = 4
    infer.q_lora_rank = 2
    infer.kv_lora_rank = 1
    infer.qk_nope_head_dim = 1
    infer.qk_rope_head_dim = 1
    infer.tp_q_head_num_ = 1
    infer.eps_ = 1e-6
    infer.use_gated_mla = False
    infer.alloc_tensor = torch.empty
    infer._tpsp_allgather = lambda input, infer_state: input

    qkv_a = _Linear(
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    q_b = _Linear(torch.eye(2))
    layer_weight = SimpleNamespace(
        qkv_a_proj_with_mqa_=qkv_a,
        q_a_layernorm_=_IdentityNorm(),
        q_b_proj_=q_b,
        kv_a_layernorm_=_IdentityNorm(),
    )
    infer_state = SimpleNamespace(need_dp_prefill_balance=False, mla_output_gate=None)
    hidden_states = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    query, cache_kv = infer._get_qkv(hidden_states, infer_state, layer_weight)

    torch.testing.assert_close(query, torch.tensor([[[1.0, 2.0]]]))
    torch.testing.assert_close(cache_kv, torch.tensor([[[3.0, 4.0]]]))


def test_kda_full_rank_output_gate_is_used_directly():
    infer = object.__new__(KimiLinearTransformerLayerInfer)
    infer.embed_dim_ = 2
    infer.kda_head_dim = 1
    infer.kda_projection_size = 2
    infer.eps_ = 1e-6
    infer.layer_num_ = 0
    infer._tpsp_allgather = lambda input, infer_state: input
    infer._kda_prefill_wrapper = lambda *args: torch.tensor([[5.0, 6.0]])

    gate_proj = _Linear(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    class _GateNorm:
        def __init__(self):
            self.gate = None

        def __call__(self, input, gate, eps):
            self.gate = gate.clone()
            return input

    gate_norm = _GateNorm()
    layer_weight = SimpleNamespace(
        kda_qkv_proj=_Linear(torch.zeros((2, 6))),
        kda_f_a_proj=_Linear(torch.ones((2, 1))),
        kda_f_b_proj=_Linear(torch.ones((1, 2))),
        kda_b_proj=_Linear(torch.ones((2, 2))),
        use_full_rank_gate=True,
        kda_g_proj=gate_proj,
        kda_o_norm=gate_norm,
        kda_o_proj=_OutputProjection(),
    )
    infer_state = SimpleNamespace(
        mem_manager=object.__new__(KimiLinearMemManager),
        req_manager=SimpleNamespace(get_mamba_cache=lambda layer_num: (None, None)),
    )
    hidden_states = torch.tensor([[1.0, 2.0]])

    output = infer._kda_forward(hidden_states, infer_state, layer_weight, is_prefill=True)

    torch.testing.assert_close(gate_proj.inputs[0], hidden_states)
    torch.testing.assert_close(gate_norm.gate, torch.tensor([[7.0], [10.0]]))
    torch.testing.assert_close(output, torch.tensor([[5.0, 6.0]]))
