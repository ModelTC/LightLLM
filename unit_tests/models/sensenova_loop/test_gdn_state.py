from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention.api import AttentionPhase
from lightllm.common.attention.api.inputs import GDNInputs, GDNParams
from lightllm.common.attention.backends.gdn.triton.backend import TritonGDNBackend
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, GDNState
from lightllm.models.sensenova_loop.config import get_loop_layer_indices, get_loop_layer_types
from lightllm.server.core.objs.start_args_type import StartArgs
from unit_tests.common.attention.helpers import make_batch, make_context, make_layer_attention, make_spec, model_stub


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mtp_step", [0, 2])
@pytest.mark.parametrize(
    "start,end,count,logical_layer,storage_layer",
    [(4, 28, 1, 4, 3), (4, 28, 3, 28, 21), (1, 6, 2, 13, 10)],
    ids=["no-loop", "aligned-loop", "unaligned-loop"],
)
def test_gdn_decode_keeps_logical_layer_state(
    monkeypatch, device, mtp_step, start, end, count, logical_layer, storage_layer
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(1585)
    settings = dict(start=start, end=end, eval_count=count, use_gate=False)
    config = dict(
        num_hidden_layers=32,
        full_attention_interval=4,
        layer_types=["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(32)],
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
    )
    indices = get_loop_layer_indices(settings, 32)
    kinds = get_loop_layer_types(settings, config)
    layers = tuple(i for i, kind in enumerate(kinds) if kind == "linear_attention")
    args = StartArgs(mtp_step=mtp_step, linear_att_ssm_data_type="float32")
    monkeypatch.setattr("lightllm.common.attention.backends._shared.linear.gdn.get_env_start_args", lambda: args)
    model = model_stub(mtp_step=mtp_step, linear_att_ssm_data_type="float32")
    model.config = config
    model.mtp_manager = SimpleNamespace(get_decode_draft_step=lambda _: mtp_step)
    backend = TritonGDNBackend(model, make_spec(use_mtp=mtp_step > 0))
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    layout = ConvSSMStateConfig(layers, 1, 2, 32, 32, 3, dtype, torch.float32)
    resource = GDNState(layout, request_capacity=3, mtp_step=mtp_step, device=device)
    resources = AttentionStateManager(request_capacity=3)
    resources.register_state("main", resource)
    resources.finish_state_registration()
    conv_view, ssm_view = resource.get_layer_view(logical_layer_id=logical_layer)
    assert conv_view.data_ptr() == resource.conv_state[storage_layer].data_ptr()
    assert ssm_view.data_ptr() == resource.ssm_state[storage_layer].data_ptr()
    physical_weights = [
        SimpleNamespace(
            layer_num_=i,
            linear_conv1d=SimpleNamespace(
                mm_param=SimpleNamespace(weight=torch.randn(layout.conv_channels, 4, device=device, dtype=dtype) * 0.1),
                bias=None,
            ),
            linear_A_log=SimpleNamespace(weight=torch.zeros(2, device=device)),
            linear_dt_bias=SimpleNamespace(weight=torch.zeros(2, device=device)),
        )
        for i in range(32)
    ]
    execution_weights = [physical_weights[i] for i in indices]
    weight = execution_weights[logical_layer]
    assert weight is physical_weights[indices[logical_layer]]

    if device == "cpu":

        def conv_pack(mixed, z, a, b, conv, *args):
            conv[1].add_(1)
            return (*backend._rearrange_mixed_qkv(mixed, decode=True), z, a, b)

        def conv_mtp(mixed, conv, *args, **kwargs):
            conv[1].add_(1)
            return mixed

        def recurrent(*, initial_state, v, **kwargs):
            initial_state[mtp_step + 1 : 2 * (mtp_step + 1)].add_(1)
            return v, None

        monkeypatch.setattr(
            "lightllm.common.attention.backends.gdn.triton.backend.conv_pack_gdn_decode_inputs", conv_pack
        )
        monkeypatch.setattr(
            "lightllm.common.attention.backends._shared.linear.causal_conv1d_mtp.causal_conv1d_update", conv_mtp
        )
        monkeypatch.setattr(
            "lightllm.common.attention.backends.gdn.triton.backend.fused_recurrent_gated_delta_rule", recurrent
        )
        monkeypatch.setattr(
            "lightllm.common.attention.backends.gdn.triton.backend.mtp_fused_recurrent_gated_delta_rule",
            recurrent,
        )

    params = GDNParams(
        conv_view,
        ssm_view,
        weight.linear_conv1d.mm_param.weight,
        None,
        weight.linear_A_log.weight,
        weight.linear_dt_bias.weight,
        mtp_state_index=resource.mtp_state_index,
    )
    mixed_dim = backend.tp_qkvz_dim + backend.tp_ba_dim
    if device == "cuda":
        batch = make_batch(
            (5,),
            device=device,
            phase=AttentionPhase.PREFILL,
            b_req_idx=torch.tensor([1], device=device, dtype=torch.int32),
            token_num=5,
            max_q_seq_len=5,
            b_q_seq_len=torch.tensor([5], device=device),
            b1_cu_q_seq_len=torch.tensor([0, 5], device=device, dtype=torch.int32),
        )
        context = make_context(backend, batch)
        backend.update(context)
        attention = replace(make_layer_attention(backend, context, params), logical_layer_id=logical_layer)
        attention.backend.forward(attention, context, GDNInputs(torch.randn(5, mixed_dim, device=device, dtype=dtype)))
    batch_size = mtp_step + 1
    batch = make_batch(
        (1,) * batch_size, device=device, b_req_idx=torch.ones(batch_size, device=device, dtype=torch.int32)
    )
    context = make_context(backend, batch)
    attention = replace(make_layer_attention(backend, context, params), logical_layer_id=logical_layer)
    untouched = [i for i in range(len(layers)) if i != storage_layer]
    for _ in range(2):
        before_conv, before_ssm = resource.conv_state.clone(), resource.ssm_state.clone()
        backend.update(context)
        output, _ = attention.backend.forward(
            attention, context, GDNInputs(torch.randn(batch_size, mixed_dim, device=device, dtype=dtype))
        )
        assert torch.isfinite(output).all()
        assert not torch.equal(resource.conv_state[storage_layer], before_conv[storage_layer])
        assert not torch.equal(resource.ssm_state[storage_layer], before_ssm[storage_layer])
        assert torch.equal(resource.conv_state[untouched], before_conv[untouched])
        assert torch.equal(resource.ssm_state[untouched], before_ssm[untouched])
        assert weight.layer_num_ == indices[logical_layer]
