from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.full_att import GQAAttentionConfig, GQAFullAttentionState
from lightllm.common.basemodel.attention.base_att import BaseAttBackend
from lightllm.common.basemodel.attention.legacy import bind_model_attention, describe_gqa_format
from lightllm.common.basemodel.attention.planner import probe_attention
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer


def bind_test_model(states, backend=None, role="main"):
    class Model:
        pass

    model = Model()
    model.data_type = torch.bfloat16
    model.att_state_role = role
    model.att_state_manager = AttentionStateManager(request_capacity=2)
    model.layers_infer = []
    for state in states:
        model.att_state_manager.register_state(role, state)
        for group in state.config.logical_layer_groups:
            for logical_layer in group:
                layer = LlamaTransformerLayerInfer.__new__(LlamaTransformerLayerInfer)
                layer.layer_num_ = logical_layer
                layer.tp_q_head_num_ = 2 * state.config.head_num
                layer.tp_k_head_num_ = state.config.head_num
                layer.head_dim_ = state.config.head_dim
                model.layers_infer.append(layer)
    model.att_state_manager.finish_state_registration()
    model.prefill_att_backend = model.decode_att_backend = backend or BaseAttBackend(model)
    model.prefill_att_backend1 = model.decode_att_backend1 = None
    bind_model_attention(model)
    return model


def make_state(groups=((3, 7),), heads=2, dim=16, device="cpu"):
    state = GQAFullAttentionState(GQAAttentionConfig(groups, heads, dim, torch.bfloat16), device=device)
    state.allocate_storage(8)
    return state


def test_bound_execution_uses_native_views_and_never_queries_state(monkeypatch):
    state = make_state()
    model = bind_test_model([state])
    owner, consumer = model.layers_infer
    assert owner.attention.spec.resource == consumer.attention.spec.resource
    assert owner.attention.cache[0].data_ptr() == consumer.attention.cache[0].data_ptr()

    def fail(*args, **kwargs):
        pytest.fail("execution looked up a state or rebuilt a view")

    monkeypatch.setattr(model.att_state_manager, "get_state", fail)
    monkeypatch.setattr(state, "get_layer_view", fail)
    calls = []

    def run(**kwargs):
        assert kwargs["k"] is consumer.attention.cache[0]
        assert kwargs["v"] is consumer.attention.cache[1]
        calls.append(kwargs["att_control"].logical_layer_id)
        return kwargs["q"]

    metadata = (SimpleNamespace(prefill_att=run, decode_att=run), None)
    q = torch.ones(2, 4, 16)
    infer = SimpleNamespace(attention_metadata=metadata)
    consumer.alloc_tensor = torch.empty
    torch.testing.assert_close(consumer._context_attention_kernel(q, None, infer, None), q)
    torch.testing.assert_close(consumer._token_attention_kernel(q, infer, None), q)
    assert calls == [7, 7]
    consumer.write_full_att_kv(token_indices=None, kv=None)
    with pytest.raises(ValueError, match="second cache write"):
        consumer.write_full_att_kv(token_indices=None, kv=q)


def test_roles_share_no_logical_resource_or_storage():
    main = bind_test_model([make_state()])
    draft = bind_test_model([make_state()], role="draft0")
    a, b = main.layers_infer[0].attention, draft.layers_infer[0].attention
    assert a.spec.resource != b.spec.resource
    assert a.cache[0].data_ptr() != b.cache[0].data_ptr()


def test_probe_rejects_mismatched_shape_and_encoding_before_binding():
    state = make_state()
    model = bind_test_model([state])
    spec = model.layers_infer[0].attention.spec
    format = describe_gqa_format(state)
    with pytest.raises(ValueError, match="shape"):
        probe_attention(model.prefill_att_backend, replace(spec, head_dim=32), format, "prefill")
    with pytest.raises(ValueError, match="unsupported KV encoding"):
        probe_attention(model.prefill_att_backend, spec, replace(format, encoding="new_int4"), "decode")


def test_flashinfer_specializes_metadata_by_actual_layer_shape(monkeypatch):
    from lightllm.common.basemodel.attention.flashinfer.fp import FlashInferAttBackend

    class ShapeBackend(FlashInferAttBackend):
        def runtime_memory_demands(self):
            return ()

        def __init__(self, model, attention_shape=None):
            self.model = model
            self.tp_q_head_num, self.tp_kv_head_num, self.head_dim = attention_shape or (4, 2, 16)

    backend = ShapeBackend(object())
    model = bind_test_model([make_state(((0,),)), make_state(((1,),), heads=4, dim=32)], backend)
    a, b = model.layers_infer
    assert a.attention.prefill_slot != b.attention.prefill_slot
    selected = model.attention_prefill_backends[b.attention.prefill_slot]
    assert (selected.tp_q_head_num, selected.tp_kv_head_num, selected.head_dim) == (8, 4, 32)
    assert model.attention_decode_backends[b.attention.decode_slot] is selected


def test_graph_update_rejects_layout_change_before_copying_any_buffer():
    from lightllm.common.basemodel.attention.flashinfer.fp import FlashInferDecodeAttState

    old = FlashInferDecodeAttState(
        kv_last_page_len_buffer=torch.zeros(2), kv_indices=torch.zeros(3), kv_starts=torch.zeros(3)
    )
    new = FlashInferDecodeAttState(
        kv_last_page_len_buffer=torch.ones(2), kv_indices=torch.ones(4), kv_starts=torch.ones(3)
    )
    with pytest.raises(ValueError, match="changed layout"):
        old.copy_for_decode_cuda_graph(new)
    assert not old.kv_last_page_len_buffer.any()


def test_workspace_lanes_do_not_alias(monkeypatch):
    from lightllm.common.basemodel.attention import base_att

    monkeypatch.setattr(base_att, "get_current_device_id", lambda: 0)
    monkeypatch.setattr(base_att.torch, "empty", lambda *args, **kwargs: object())
    monkeypatch.setattr(BaseAttBackend, "_workspace_buffers", {})
    lane0 = BaseAttBackend.get_gpu_workspace_buffer("test", 128, lane_id=0)
    assert lane0 is BaseAttBackend.get_gpu_workspace_buffer("test", 128, lane_id=0)
    assert lane0 is not BaseAttBackend.get_gpu_workspace_buffer("test", 128, lane_id=1)


def test_joint_choice_uses_one_representation_for_both_phases():
    from lightllm.common.basemodel.attention.planner import select_attention

    model = bind_test_model([make_state()])
    spec, plain = model.layers_infer[0].attention.spec, describe_gqa_format(make_state())
    alternate = replace(plain, encoding="alternate")

    class AlternateBackend(BaseAttBackend):
        supported_kv_encodings = ("alternate",)

    plain_backend, alternate_backend = BaseAttBackend(object()), AlternateBackend(object())
    candidates = (
        lambda format, phase: (plain_backend, alternate_backend) if phase == "prefill" else (alternate_backend,)
    )
    choice = select_attention(spec, (plain, alternate), candidates)
    assert choice.format is alternate
    assert choice.prefill.backend is choice.decode.backend is alternate_backend
    assert any("no decode implementation for plain" in reason for reason in choice.rejected)


def test_explicit_incompatible_backend_explains_layer_and_constraints():
    from lightllm.common.basemodel.attention.planner import select_attention

    model = bind_test_model([make_state()])
    spec = replace(model.layers_infer[0].attention.spec, attention_sink=True)
    format = describe_gqa_format(make_state())
    with pytest.raises(ValueError, match="main layer 3.*unsupported attention feature sink"):
        select_attention(spec, (format,), lambda format, phase: (model.prefill_att_backend,))


@pytest.mark.parametrize("change", [dict(layout="interleaved"), dict(version=2)])
def test_provider_rejects_unknown_byte_layout_before_binding(change):
    model = bind_test_model([make_state()])
    spec = model.layers_infer[0].attention.spec
    with pytest.raises(ValueError, match="unsupported KV layout"):
        probe_attention(model.prefill_att_backend, spec, replace(describe_gqa_format(make_state()), **change), "decode")


def test_triton_rejects_speculative_window_before_execution():
    from lightllm.common.basemodel.attention.triton.fp import TritonAttBackend

    model = bind_test_model([make_state()])
    model.mtp_manager = SimpleNamespace(get_decode_draft_step=lambda _: 3)
    model.is_mtp_draft_model = False
    spec = replace(model.layers_infer[0].attention.spec, window=(128, 0))
    with pytest.raises(ValueError, match="speculative decode does not implement sliding windows"):
        probe_attention(TritonAttBackend(model), spec, describe_gqa_format(make_state()), "decode")


def test_mla_binds_heterogeneous_projection_shapes_before_allocation():
    from lightllm.common.attention_state.full_att import MLAConfig, MLAFullAttentionState
    from lightllm.common.basemodel.attention.flashinfer.mla import MlaFlashInferAttBackend
    from lightllm.models.deepseek2.layer_infer.transformer_layer_infer import Deepseek2TransformerLayerInfer

    class Model:
        pass

    class ShapeBackend(MlaFlashInferAttBackend):
        def runtime_memory_demands(self):
            return ()

        def __init__(self, model, attention_shape=None):
            self.model = model
            (
                self.tp_q_head_num,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.kv_lora_rank,
                self.v_head_dim,
                self.q_data_type,
                self.softmax_scale,
            ) = attention_shape or (4, 128, 64, 512, 128, torch.bfloat16, 0.125)

    model = Model()
    model.data_type, model.att_state_role = torch.bfloat16, "main"
    model.att_state_manager = AttentionStateManager(request_capacity=1)
    model.layers_infer = []
    for i, latent in enumerate((512, 256)):
        state = MLAFullAttentionState(MLAConfig(((i,),), latent + 64, torch.bfloat16), device="cpu")
        state.allocate_storage(8)
        model.att_state_manager.register_state("main", state)
        layer = Deepseek2TransformerLayerInfer.__new__(Deepseek2TransformerLayerInfer)
        layer.layer_num_, layer.tp_q_head_num_ = i, 4 * (i + 1)
        layer.qk_nope_head_dim, layer.qk_rope_head_dim = 128, 64
        layer.kv_lora_rank, layer.v_head_dim, layer.softmax_scale = latent, 128, 0.125
        model.layers_infer.append(layer)
    model.att_state_manager.finish_state_registration()
    model.prefill_att_backend = model.decode_att_backend = ShapeBackend(model)
    model.prefill_att_backend1 = model.decode_att_backend1 = None
    bind_model_attention(model)
    first, second = [layer.attention for layer in model.layers_infer]
    assert first.resource != second.resource
    assert first.prefill_slot != second.prefill_slot
    backend = model.attention_decode_backends[second.decode_slot]
    assert (backend.tp_q_head_num, backend.kv_lora_rank) == (8, 256)
    assert second.cache.shape[-1] == 320
