from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state.bounded_att import ConvSSMStateConfig, GDNState, KDAState
from lightllm.common.attention_state.commit import RequestGenerationStamp
from lightllm.common.attention_state.manager import AttentionStateManager


def test_commit_owns_mapping_and_detects_recycled_requests():
    requests = torch.tensor([2, 2, 4], dtype=torch.int32)
    steps = torch.tensor([0, 1, 0], dtype=torch.int32)
    generations = torch.zeros(8, dtype=torch.int64)
    output = RequestGenerationStamp.stamp("main", requests, steps, generations)
    requests.fill_(6)
    steps.fill_(5)
    assert output.request_indices.tolist() == [2, 2, 4]
    assert output.mtp_indices.tolist() == [0, 1, 0]
    output.validate_generation(generations)
    generations[2] += 1
    with pytest.raises(RuntimeError, match="reused request slot"):
        output.validate_generation(generations)


def test_overlap_commit_preserves_microbatch_order_and_role():
    generation = torch.zeros(8, dtype=torch.int64)
    a = RequestGenerationStamp.stamp("main", torch.tensor([4, 4]), torch.tensor([0, 1]), generation)
    b = RequestGenerationStamp.stamp("main", torch.tensor([2]), torch.tensor([0]), generation)
    merged = RequestGenerationStamp.concatenate((a, b))
    assert merged.request_indices.tolist() == [4, 4, 2]
    assert merged.mtp_indices.tolist() == [0, 1, 0]
    with pytest.raises(ValueError, match="same model role"):
        RequestGenerationStamp.concatenate((a, replace(b, role="draft0")))


@pytest.mark.parametrize("state_cls", [GDNState, KDAState])
def test_capture_snapshot_restores_only_participating_role_and_requests(state_cls):
    config = ConvSSMStateConfig(
        logical_layer_ids=(0,),
        global_k_heads=1,
        global_v_heads=1,
        key_dim=2,
        value_dim=2,
        conv_width=2,
        conv_dtype=torch.bfloat16,
        ssm_dtype=torch.float32,
    )
    manager = AttentionStateManager(request_capacity=3)
    main = state_cls(config, request_capacity=3, mtp_step=2, device="cpu")
    draft = state_cls(config, request_capacity=3, mtp_step=2, device="cpu")
    manager.register_state("main", main)
    manager.register_state("draft0", draft)
    manager.finish_state_registration()
    main.conv_state.fill_(2)
    main.ssm_state.fill_(3)
    main.mtp_state_index.fill_(1)
    snapshot = manager.snapshot_capture_state(torch.tensor([1, 1]), role="main")
    main.conv_state[:, 1].fill_(8)
    main.ssm_state[:, 3:6].fill_(9)
    main.mtp_state_index[1] = 2
    main.conv_state[:, 2].fill_(6)
    draft.conv_state.fill_(7)
    manager.restore_capture_state(snapshot)
    assert torch.all(main.conv_state[:, 1] == 2)
    assert torch.all(main.ssm_state[:, 3:6] == 3)
    assert main.mtp_state_index[1] == 1
    assert torch.all(main.conv_state[:, 2] == 6)
    assert torch.all(draft.conv_state == 7)


@pytest.mark.parametrize("kind", ["mla", "packed_bf16", "packed_group128_fp8", "packed_group64_e8m0"])
def test_mla_adapter_binds_state_layout(kind):
    from lightllm.common.attention.api import AttentionFamily, AttentionPhase
    from lightllm.common.attention_state.full_att import (
        MLAConfig,
        MLAFullAttentionState,
        PackedMLALogicalConfig,
        create_packed_mla_state,
    )
    from lightllm.common.basemodel.attention_adapters import ADAPTERS, AttentionDescription
    from unit_tests.common.attention.helpers import model_stub

    nope = kind == "packed_group64_e8m0"
    rope_dim = 0 if nope else 64
    latent_dim = 512 + rope_dim
    if kind == "mla":
        state = MLAFullAttentionState(MLAConfig(((2, 5),), latent_dim, torch.bfloat16), device="cpu")
    else:
        state = create_packed_mla_state(PackedMLALogicalConfig(((2, 5),), latent_dim, kind), device="cpu")
    manager = AttentionStateManager(request_capacity=2)
    manager.register_state("main", state)
    manager.finish_state_registration()
    model = model_stub(llm_kv_type="fp8kv_dsa" if "fp8" in kind or nope else "None")
    model.att_state_manager, model.data_type = manager, torch.bfloat16
    layer = SimpleNamespace(
        layer_num_=5,
        tp_q_head_num_=16,
        qk_nope_head_dim=512,
        kv_lora_rank=512,
        v_head_dim=128,
        softmax_scale=latent_dim ** -0.5,
        network_config_={} if nope else {"qk_rope_head_dim": rope_dim},
    )
    if not nope:
        layer.qk_rope_head_dim = rope_dim
    family = AttentionFamily.MLA if kind == "mla" else AttentionFamily.SPARSE_MLA
    adapter = ADAPTERS[family]
    spec = adapter.describe(model, layer, AttentionPhase.PREFILL, AttentionDescription(family))
    assert state.storage is None
    state.allocate_storage(8)
    resources = adapter.bind(model, layer, None)
    assert spec.cache.dtype == resources.kv.dtype == state.storage_dtype
    assert resources.qk_rope_head_dim == rope_dim
    assert resources.kv.data_ptr() == state.layer_resources(2).data_ptr()
    assert resources.kv.shape[-1] == (584 if nope else 656 if kind == "packed_group128_fp8" else latent_dim)


@pytest.mark.parametrize("family", ["gdn", "kda"])
def test_replay_view_has_no_lifecycle_methods(family):
    from importlib import import_module

    state_cls = getattr(
        import_module(f"lightllm.common.attention_state.bounded_att.{family}.replay_{family}_state"),
        f"Replay{family.upper()}State",
    )
    config = ConvSSMStateConfig((4, 1), 1, 1, 8, 8, 3, torch.float32, torch.float32)
    state = state_cls(config, request_capacity=3, device="cpu")
    resources = state.layer_resources(1)
    replay = resources.replay_state
    assert replay is not state
    assert replay.ssm_state is state.ssm_state
    assert replay.cursors is state.cursors
    assert replay.storage_layer(1) == 1
    assert resources.ssm.data_ptr() == state.ssm_state[1].data_ptr()
    assert not any(
        hasattr(replay, name) for name in ("finish_decode", "finish_decode_token", "reset_bounded_att_state")
    )


@pytest.mark.parametrize("failure", [False, True])
def test_model_finishes_replay_once_after_all_layers(failure):
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.common.basemodel.attention import ModelAttention
    from unit_tests.common.attention.helpers import make_batch

    events = []
    coordinator = ModelAttention.__new__(ModelAttention)
    coordinator.draft_step, coordinator.dynamic_verify = 0, False
    coordinator.replay_states = (
        SimpleNamespace(finish_decode=lambda reqs, cu: events.append(("finish", reqs.tolist(), cu.tolist()))),
    )
    state = SimpleNamespace(
        attention=SimpleNamespace(batch=make_batch(), build_device_metadata=lambda: None),
        input_ids=torch.ones(2, 4),
        is_cuda_graph_capture=False,
        hidden_collector=SimpleNamespace(add=lambda **kw: None, add_final_hidden=lambda *a: None),
    )

    def layer(index):
        def forward(hidden, *args):
            events.append(index)
            if failure and index == 1:
                raise RuntimeError("layer failed")
            return hidden

        return SimpleNamespace(token_forward=forward)

    model = SimpleNamespace(
        attention=coordinator,
        layers_num=2,
        layers_infer=[layer(4), layer(1)],
        trans_layers_weight=[None, None],
        pre_post_weight=None,
        pre_infer=SimpleNamespace(token_forward=lambda x, *a: x, _tpsp_sp_split=lambda input, **kw: input),
        post_infer=SimpleNamespace(token_forward=lambda x, **kw: x, _tpsp_allgather=lambda input, **kw: input),
        _create_model_output=lambda x, state: x,
    )
    if failure:
        with pytest.raises(RuntimeError, match="layer failed"):
            TpPartBaseModel._token_forward(model, state)
        assert events == [4, 1]
    else:
        TpPartBaseModel._token_forward(model, state)
        assert events == [4, 1, ("finish", [0, 1], [0, 1, 2])]


def test_glm_query_metadata_is_owned_by_model(monkeypatch):
    from lightllm.common.basemodel.infer_struct import InferStateInfo
    from lightllm.models.glm5_next.infer_struct import Glm5NextInferStateInfo
    from unit_tests.common.attention.helpers import make_batch

    monkeypatch.setattr(InferStateInfo, "init_some_extra_state", lambda *a: None)
    captured, incoming = Glm5NextInferStateInfo(), Glm5NextInferStateInfo()
    table = make_batch().req_to_token_indexs
    for state, positions in ((captured, [3, 4, 3]), (incoming, [5, 6, 7])):
        state.is_prefill = True
        state.position_ids = torch.tensor(positions, dtype=torch.int32)
        state.b_req_idx = torch.tensor([0, 1], dtype=torch.int32)
        state.b_q_seq_len = torch.tensor([2, 1], dtype=torch.int32)
        state.input_ids = torch.zeros(3, dtype=torch.int64)
        state.init_some_extra_state(None)
        state.attention.batch = make_batch(req_to_token_indexs=table)
    assert captured.sparse_queries.q_kv_seq_len.tolist() == [4, 5, 4]
    assert captured.sparse_queries.q_req_idx.tolist() == [0, 0, 1]
    pointer = captured.q_kv_seq_len.data_ptr()
    for target, source in captured.graph_input_copies(incoming):
        target.copy_(source)
    assert captured.q_kv_seq_len.data_ptr() == pointer
    assert captured.sparse_queries.q_kv_seq_len.tolist() == [6, 7, 8]
