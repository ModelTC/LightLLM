from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention import selector
from lightllm.common.attention.backends.gdn import GDNLayerConfig
from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, GDNState
from lightllm.common.basemodel.attention_runtime import AttentionStepConfig, configure_attention_step
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import Qwen3NextTransformerLayerInfer
from lightllm.models.sensenova_loop.config import get_loop_layer_indices, get_loop_layer_types
from lightllm.server.core.objs.start_args_type import StartArgs


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
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    # The GDN layers' attention: Triton prefill and decode over native state.
    layer_config = GDNLayerConfig(1, 2, 32, 32, 4, "silu", dtype, state_mtp_step=mtp_step)
    prefill_selection = selector.select(layer_config, "prefill", "triton")
    decode_selection = selector.select(layer_config, "decode", "triton")
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

        def conv_pack(mixed, a, b, conv, *args):
            conv[1].add_(1)
            q, k, v = mixed.split([32, 32, 64], dim=-1)
            rows = mixed.shape[0]
            return q.view(rows, 1, 1, 32), k.view(rows, 1, 1, 32), v.view(rows, 1, 2, 32), a, b

        def conv_mtp(mixed, conv, *args, **kwargs):
            conv[1].add_(1)
            return mixed

        def recurrent(*, initial_state, v, out, **kwargs):
            # Like the kernels, write the output into the caller's out.
            initial_state[mtp_step + 1 : 2 * (mtp_step + 1)].add_(1)
            out.copy_(v)
            return out, None

        monkeypatch.setattr(
            "lightllm.common.attention.kernels.gdn.gdn_decode_pack.conv_pack_gdn_decode_inputs", conv_pack
        )
        monkeypatch.setattr("lightllm.common.attention.kernels.gdn.causal_conv1d_mtp.causal_conv1d_update", conv_mtp)
        monkeypatch.setattr(
            "lightllm.common.attention.kernels.gdn.third_party_fla.ops.fused_recurrent."
            "fused_recurrent_gated_delta_rule",
            recurrent,
        )
        monkeypatch.setattr(
            "lightllm.common.attention.kernels.gdn.mtp_fused_recurrent.mtp_fused_recurrent_gated_delta_rule",
            recurrent,
        )

    infer = InferStateInfo()
    infer.att_state_manager, infer.att_state_role = resources, "main"
    infer.b_req_idx = torch.tensor([1], device=device, dtype=torch.int32)
    infer.req_manager = SimpleNamespace(req_to_token_indexs=torch.zeros(3, 8, device=device, dtype=torch.int32))
    # The model layer resolves its state views and parameters by logical layer.
    layer = Qwen3NextTransformerLayerInfer.__new__(Qwen3NextTransformerLayerInfer)
    layer.layer_num_, layer.tp_linear_num_v_heads, layer.linear_head_v_dim, layer.tp_linear_key_dim = (
        logical_layer,
        2,
        32,
        32,
    )
    layer.prefill_impl, layer.decode_impl = prefill_selection.impl, decode_selection.impl
    layer.alloc_tensor = lambda shape, dtype=None, device=None, **_: torch.empty(shape, dtype=dtype, device=device)
    mixed_dim = 2 * 32 + 2 * 64 + 2 * 2

    def build_attention_states(backend):
        # The attention state the layers' group builds for this step.
        step_config = AttentionStepConfig(
            decode_rows_per_request=mtp_step + 1, req_to_mtp_state_index=resource.mtp_state_index
        )
        batch = configure_attention_step(infer, step_config)
        impl = layer.prefill_impl if infer.is_prefill else layer.decode_impl
        infer.att_states = {impl.state_key: backend.build(batch, backend.create_build_context(impl.state_key[1], None))}

    if device == "cuda":
        infer.b1_cu_q_seq_len = torch.tensor([0, 5], device=device, dtype=torch.int32)
        infer.b_ready_cache_len = torch.zeros(1, device=device, dtype=torch.int32)
        infer.is_prefill = True
        build_attention_states(prefill_selection.backend)
        inputs = layer._gdn_layer_inputs(torch.randn(5, mixed_dim, device=device, dtype=dtype), infer, weight)
        layer._gdn_attention(inputs, infer, prefill=True)

    infer.batch_size = mtp_step + 1
    infer.b_req_idx = infer.b_req_idx.repeat(infer.batch_size)
    infer.is_prefill = False
    infer.b_seq_len = torch.ones_like(infer.b_req_idx)
    untouched = [i for i in range(len(layers)) if i != storage_layer]
    for _ in range(2):
        before_conv, before_ssm = resource.conv_state.clone(), resource.ssm_state.clone()
        build_attention_states(decode_selection.backend)
        inputs = layer._gdn_layer_inputs(
            torch.randn(infer.batch_size, mixed_dim, device=device, dtype=dtype), infer, weight
        )
        output = layer._gdn_attention(inputs, infer, prefill=False)
        assert torch.isfinite(output).all()
        assert not torch.equal(resource.conv_state[storage_layer], before_conv[storage_layer])
        assert not torch.equal(resource.ssm_state[storage_layer], before_ssm[storage_layer])
        assert torch.equal(resource.conv_state[untouched], before_conv[untouched])
        assert torch.equal(resource.ssm_state[untouched], before_ssm[untouched])
        assert weight.layer_num_ == indices[logical_layer]
