from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, GDNState
from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.attention.linear.triton import TritonLinearAttBackend
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.sensenova_loop.config import get_loop_layer_indices, get_loop_layer_types
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mtp_step", [0, 2])
@pytest.mark.parametrize("cache_mode", ["independent", "overwrite"])
@pytest.mark.parametrize(
    "start,end,count,execution_layer,independent_slot,overwrite_slot",
    [(4, 28, 1, 4, 3, 3), (4, 28, 3, 28, 21, 3), (1, 6, 2, 13, 10, 6)],
    ids=["no-loop", "aligned-loop", "unaligned-loop"],
)
def test_gdn_loop_visits_update_selected_layer_state(
    monkeypatch, device, mtp_step, cache_mode, start, end, count, execution_layer, independent_slot, overwrite_slot
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
    physical_layer = indices[execution_layer]
    logical_layer = physical_layer if cache_mode == "overwrite" else execution_layer
    storage_layer = overwrite_slot if cache_mode == "overwrite" else independent_slot
    kinds = config["layer_types"] if cache_mode == "overwrite" else get_loop_layer_types(settings, config)
    layers = tuple(i for i, kind in enumerate(kinds) if kind == "linear_attention")
    args = StartArgs(mtp_step=mtp_step, linear_att_ssm_data_type="float32")
    monkeypatch.setattr("lightllm.common.basemodel.attention.linear.gdn.get_env_start_args", lambda: args)
    monkeypatch.setattr("lightllm.common.basemodel.attention.base_att.get_env_start_args", lambda: args)
    backend = object.__new__(TritonLinearAttBackend)
    backend.__init__(
        SimpleNamespace(
            config=config,
            tp_world_size_=1,
            is_mtp_draft_model=False,
            mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: mtp_step),
        )
    )
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
    weight = execution_weights[execution_layer]
    assert weight is physical_weights[physical_layer]

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

        monkeypatch.setattr("lightllm.common.basemodel.attention.linear.gdn.conv_pack_gdn_decode_inputs", conv_pack)
        monkeypatch.setattr(
            "lightllm.common.basemodel.triton_kernel.linear_att.causal_conv1d_mtp.causal_conv1d_update", conv_mtp
        )
        monkeypatch.setattr(
            "lightllm.common.basemodel.attention.linear.gdn.fused_recurrent_gated_delta_rule", recurrent
        )
        monkeypatch.setattr(
            "lightllm.common.basemodel.attention.linear.gdn.mtp_fused_recurrent_gated_delta_rule",
            recurrent,
        )

    infer = InferStateInfo()
    infer.att_state_manager, infer.att_state_role = resources, "main"
    infer.b_req_idx = torch.tensor([1], device=device, dtype=torch.int32)
    control = dict(layer_num=logical_layer, layer_weight=weight)
    mixed_dim = backend.tp_qkvz_dim + backend.tp_ba_dim
    if device == "cuda":
        infer.b1_cu_q_seq_len = torch.tensor([0, 5], device=device, dtype=torch.int32)
        infer.b_ready_cache_len = torch.zeros(1, device=device, dtype=torch.int32)
        prefill = backend.create_att_prefill_state(infer)
        prefill.init_state()
        prefill.prefill_att(
            None,
            None,
            None,
            AttControl(
                linear_att_prefill=True,
                linear_att_prefill_dict=dict(
                    **control, mixed_qkvzba=torch.randn(5, mixed_dim, device=device, dtype=dtype)
                ),
            ),
        )

    infer.batch_size = mtp_step + 1
    infer.b_req_idx = infer.b_req_idx.repeat(infer.batch_size)
    decode = backend.create_att_decode_state(infer)
    untouched = [i for i in range(len(layers)) if i != storage_layer]
    for _ in range(2):
        before_conv, before_ssm = resource.conv_state.clone(), resource.ssm_state.clone()
        decode.init_state()
        output, _ = decode.decode_att(
            None,
            None,
            None,
            AttControl(
                linear_att_decode=True,
                linear_att_decode_dict=dict(
                    **control, mixed_qkvzba=torch.randn(infer.batch_size, mixed_dim, device=device, dtype=dtype)
                ),
            ),
        )
        assert torch.isfinite(output).all()
        assert not torch.equal(resource.conv_state[storage_layer], before_conv[storage_layer])
        assert not torch.equal(resource.ssm_state[storage_layer], before_ssm[storage_layer])
        assert torch.equal(resource.conv_state[untouched], before_conv[untouched])
        assert torch.equal(resource.ssm_state[untouched], before_ssm[untouched])
        assert weight.layer_num_ == physical_layer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("phase", ["prefill", "decode", "decode_graph"])
@pytest.mark.parametrize("mtp_step", [0, 2])
@pytest.mark.parametrize("ssm_dtype", [torch.float32, torch.bfloat16])
def test_loop_visits_read_old_history_and_commit_only_last(monkeypatch, phase, mtp_step, ssm_dtype):
    torch.manual_seed(2468)
    args = StartArgs(mtp_step=mtp_step, linear_att_ssm_data_type=str(ssm_dtype).removeprefix("torch."))
    monkeypatch.setattr("lightllm.common.basemodel.attention.linear.gdn.get_env_start_args", lambda: args)
    monkeypatch.setattr("lightllm.common.basemodel.attention.base_att.get_env_start_args", lambda: args)
    config = dict(
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
    )
    backend = object.__new__(TritonLinearAttBackend)
    backend.__init__(
        SimpleNamespace(
            config=config,
            tp_world_size_=1,
            is_mtp_draft_model=False,
            mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: mtp_step),
        )
    )
    layout = ConvSSMStateConfig((4, 8), 1, 2, 32, 32, 3, torch.bfloat16, ssm_dtype)
    weight = SimpleNamespace(
        linear_conv1d=SimpleNamespace(
            mm_param=SimpleNamespace(
                weight=torch.randn(layout.conv_channels, 4, device="cuda", dtype=torch.bfloat16) * 0.1
            ),
            bias=None,
        ),
        linear_A_log=SimpleNamespace(weight=torch.zeros(2, device="cuda")),
        linear_dt_bias=SimpleNamespace(weight=torch.zeros(2, device="cuda")),
    )

    def make_runtime():
        resource = GDNState(layout, request_capacity=5, mtp_step=mtp_step, device="cuda")
        resource.mtp_state_index.fill_(min(1, mtp_step))
        resources = AttentionStateManager(request_capacity=5)
        resources.register_state("main", resource)
        resources.finish_state_registration()
        infer = InferStateInfo()
        infer.att_state_manager, infer.att_state_role = resources, "main"
        infer.b_req_idx = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
        if phase == "prefill":
            infer.b1_cu_q_seq_len = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
            infer.b_ready_cache_len = torch.tensor([2, 7], dtype=torch.int32, device="cuda")
            att = backend.create_att_prefill_state(infer)
        else:
            infer.batch_size = 2 * (mtp_step + 1)
            infer.b_req_idx = infer.b_req_idx.repeat_interleave(mtp_step + 1)
            att = backend.create_att_decode_state(infer)
        att.init_state()
        return resource, infer, att

    actual, actual_infer, actual_att = make_runtime()
    reference, reference_infer, reference_att = make_runtime()
    actual.conv_state.normal_(0, 0.1)
    actual.ssm_state.normal_(0, 0.1)
    before_conv, before_ssm = actual.conv_state.clone(), actual.ssm_state.clone()
    conv_indices = actual_att.b_conv_buffer_idx.clone()
    ssm_indices = actual_att.b_ssm_buffer_idx.clone()

    def run(att, mixed, write_state):
        # MTP convolution overwrites the qkv projection view in place.
        params = dict(layer_num=4, layer_weight=weight, mixed_qkvzba=mixed.clone(), write_state=write_state)
        if phase == "prefill":
            return att.prefill_att(
                None,
                None,
                None,
                AttControl(
                    linear_att_prefill=True,
                    linear_att_prefill_dict=params,
                ),
            )[0]
        return att.decode_att(
            None,
            None,
            None,
            AttControl(
                linear_att_decode=True,
                linear_att_decode_dict=params,
            ),
        )[0]

    rows = 5 if phase == "prefill" else actual_infer.batch_size
    mixed_dim = backend.tp_qkvz_dim + backend.tp_ba_dim
    for visit in range(3):
        mixed = torch.randn(rows, mixed_dim, dtype=torch.bfloat16, device="cuda")
        reference.conv_state.copy_(before_conv)
        reference.ssm_state.copy_(before_ssm)
        expected_output = run(reference_att, mixed, True)
        write_state = visit == 2
        if phase == "decode_graph":
            # Warm both the original-pool and compact temporary-pool paths.
            run(actual_att, mixed, write_state)
            actual.conv_state.copy_(before_conv)
            actual.ssm_state.copy_(before_ssm)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = run(actual_att, mixed, write_state)
            actual.conv_state.copy_(before_conv)
            actual.ssm_state.copy_(before_ssm)
            graph.replay()
        else:
            output = run(actual_att, mixed, write_state)
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.conv_state, reference.conv_state if write_state else before_conv, rtol=0, atol=0
        )
        torch.testing.assert_close(actual.ssm_state, reference.ssm_state if write_state else before_ssm, rtol=0, atol=0)
        assert torch.equal(actual_att.b_conv_buffer_idx, conv_indices)
        assert torch.equal(actual_att.b_ssm_buffer_idx, ssm_indices)
    assert not torch.equal(actual.conv_state, before_conv)
    assert not torch.equal(actual.ssm_state, before_ssm)
