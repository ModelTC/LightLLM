from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, GDNState, KDAState
from lightllm.common.basemodel.attention.legacy import (
    prepare_model_attention,
    bind_model_attention,
    explain_attention_plans,
)
from lightllm.common.basemodel.attention.linear.triton import TritonLinearAttBackend
from lightllm.common.basemodel.attention.linear.kda import KDALinearAttBackend
from lightllm.common.basemodel.attention.recurrent import GDNInput, KDAInput
from lightllm.common.basemodel.infer_struct import InferStateInfo


class Model:
    pass


def make_model(monkeypatch, family, device="cpu", mtp_step=0):
    monkeypatch.setattr(
        "lightllm.common.basemodel.attention.linear.gdn.get_env_start_args",
        lambda: SimpleNamespace(mtp_step=mtp_step, linear_att_ssm_data_type="float32"),
    )
    monkeypatch.setattr(
        "lightllm.common.basemodel.attention.base_att.get_env_start_args",
        lambda: SimpleNamespace(mtp_dynamic_verify=False),
    )
    model = Model()
    model.args = SimpleNamespace(mtp_step=mtp_step)
    model.tp_world_size_ = 1
    model.att_state_role = "main"
    model.mtp_manager = SimpleNamespace(get_decode_draft_step=lambda _: mtp_step)
    model.is_mtp_draft_model = False
    model.config = dict(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        linear_attn_config=dict(num_heads=1, head_dim=64, short_conv_kernel_size=4),
    )
    model.att_state_manager = AttentionStateManager(request_capacity=3)
    model.layers_infer = [SimpleNamespace(layer_num_=layer) for layer in (0, 2)]
    model.trans_layers_weight = [None, None]
    states = []
    for layer, heads, dim in ((0, 1, 64), (2, 2, 128)):
        if family == "kda":
            dim = 128
        config = ConvSSMStateConfig((layer,), heads, heads, dim, dim, 3, torch.bfloat16, torch.float32)
        state = (GDNState if family == "gdn" else KDAState)(config, 3, mtp_step, device=device)
        model.att_state_manager.register_state("main", state)
        states.append(state)
    model.att_state_manager.finish_state_registration()
    model.prefill_att_backend = model.decode_att_backend = None
    backend = (TritonLinearAttBackend if family == "gdn" else KDALinearAttBackend)(model)
    model.prefill_att_backend1 = model.decode_att_backend1 = backend
    prepare_model_attention(model)
    bind_model_attention(model)
    return model, states


def decode_metadata(model, requests):
    infer = InferStateInfo()
    infer.is_prefill = False
    infer.b_req_idx = requests
    infer.batch_size = requests.numel()
    infer.attention_metadata = tuple(
        backend.create_att_decode_state(infer) if backend is not None else None
        for backend in model.attention_decode_backends
    )
    infer.init_att_state()
    return infer


@pytest.mark.parametrize("family", ["gdn", "kda"])
def test_recurrent_domains_bind_their_own_shapes_and_accepted_cursor(monkeypatch, family):
    model, states = make_model(monkeypatch, family, mtp_step=2)
    plans = explain_attention_plans(model)
    assert [plan["layer"] for plan in plans] == [0, 2]
    assert [plan["storage_row"] for plan in plans] == [0, 0]
    assert all(plan["writer"] == "recurrent_update" for plan in plans)
    states[0].mtp_state_index.copy_(torch.tensor([0, 2, 0], dtype=torch.int32))
    states[1].mtp_state_index.copy_(torch.tensor([1, 0, 0], dtype=torch.int32))
    infer = decode_metadata(model, torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32))
    for layer, state, accepted in zip(model.layers_infer, states, ([1, 3], [2, 1])):
        binding = layer.recurrent
        metadata = infer.attention_metadata[binding.decode_slot]
        assert metadata.backend.mtp_state_index is state.mtp_state_index
        assert metadata.b_num_accepted_tokens.tolist() == accepted
        if family == "gdn":
            assert metadata.backend.tp_num_k_heads == state.config.num_k_heads
            assert metadata.backend.head_k_dim == state.config.key_dim
        else:
            assert metadata.backend.tp_num_heads == state.config.num_k_heads
            assert metadata.backend.head_dim == state.config.key_dim
    assert model.layers_infer[0].recurrent.decode_slot != model.layers_infer[1].recurrent.decode_slot


def test_kda_rejects_unsupported_shape_before_execution(monkeypatch):
    from dataclasses import replace

    model, states = make_model(monkeypatch, "kda")
    backend = model.attention_decode_backends[1]
    with pytest.raises(ValueError, match="head dimension 128"):
        backend.for_recurrent_state(replace(states[0].config, key_dim=64, value_dim=64), 0)


@pytest.mark.parametrize("family", ["gdn", "kda"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_heterogeneous_recurrent_graph_updates_only_current_requests(monkeypatch, family):
    torch.manual_seed(56)
    model, states = make_model(monkeypatch, family, device="cuda")
    inputs = []
    for i, state in enumerate(states):
        c = state.config
        weight = SimpleNamespace(
            linear_A_log=SimpleNamespace(weight=torch.randn(c.num_v_heads, device="cuda")),
            linear_dt_bias=SimpleNamespace(
                weight=torch.randn(c.num_v_heads if family == "gdn" else c.num_v_heads * c.key_dim, device="cuda")
            ),
        )
        conv = torch.randn(c.conv_channels, 4, dtype=torch.bfloat16, device="cuda")
        weight.linear_conv1d = SimpleNamespace(mm_param=SimpleNamespace(weight=conv), bias=None)
        weight.get_merged_kda_conv_weight = lambda conv=conv: conv
        model.trans_layers_weight[i] = weight
        if family == "gdn":
            width = c.conv_channels + c.num_v_heads * (c.value_dim + 2)
            inputs.append(GDNInput(torch.randn(2, width, dtype=torch.bfloat16, device="cuda")))
        else:
            inputs.append(
                KDAInput(
                    torch.randn(2, c.conv_channels, dtype=torch.bfloat16, device="cuda"),
                    torch.randn(2, c.num_v_heads * c.key_dim, dtype=torch.bfloat16, device="cuda"),
                    torch.randn(2, c.num_v_heads, dtype=torch.bfloat16, device="cuda"),
                )
            )
    bind_model_attention(model)
    infer = decode_metadata(model, torch.tensor([0, 1], dtype=torch.int32, device="cuda"))
    saved_inputs = [{name: tensor.clone() for name, tensor in vars(x).items()} for x in inputs]

    def reset():
        for x, saved in zip(inputs, saved_inputs):
            for name, tensor in saved.items():
                getattr(x, name).copy_(tensor)
        for state in states:
            state.conv_state.zero_()
            state.ssm_state.zero_()

    def run():
        return [layer.recurrent.decode(x, infer.attention_metadata) for layer, x in zip(model.layers_infer, inputs)]

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for reqs in ([1, 2], [2, 0]):
        infer.b_req_idx.copy_(torch.tensor(reqs, dtype=torch.int32, device="cuda"))
        reset()
        expected = run()
        expected_states = [(s.conv_state.clone(), s.ssm_state.clone()) for s in states]
        reset()
        graph.replay()
        for actual, reference in zip(captured, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        for state, (conv, ssm) in zip(states, expected_states):
            torch.testing.assert_close(state.conv_state, conv, rtol=0, atol=0)
            torch.testing.assert_close(state.ssm_state, ssm, rtol=0, atol=0)
            untouched = next(req for req in range(3) if req not in reqs)
            assert not state.conv_state[:, untouched].count_nonzero().item()
            assert not state.ssm_state[:, untouched].count_nonzero().item()
