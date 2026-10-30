"""按角色、层归属及方案查询资源，不在推理层保存绑定状态。"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.att_resources import AttResourceManager
from lightllm.common.att_resources.full_att import FullAttentionConfig, FullAttentionState, GQAFullAttentionState
from lightllm.common.att_resources.bounded_att import KDAState, GDNState
from unit_tests.common.att_resources.test_bounded_state import make_state
from unit_tests.common.att_resources.test_tail_state import make_tail
from unit_tests.common.att_resources.test_capacity import definition
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.common.basemodel.layer_infer.transformer_layer_infer import TransformerLayerInfer


def test_creation_does_not_bind_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    model = LlamaTpPartModel.__new__(LlamaTpPartModel)
    model.args, model.config = SimpleNamespace(), {}
    model.tp_rank_, model.tp_world_size_ = 0, 1
    model.layers_infer = [SimpleNamespace(layer_num_=0, indexer=SimpleNamespace())]
    model.max_req_num = 0
    model.torch_memory_saver = SimpleNamespace(region=lambda **_: nullcontext())
    state = GQAFullAttentionState(
        layout_config=FullAttentionConfig(
            logical_layer_groups=((0,),),
            global_head_num=1,
            head_dim=128,
            dtype=torch.bfloat16,
        ), device="meta",
    )
    monkeypatch.setattr(
        LlamaTpPartModel, "_create_full_att_states", classmethod(lambda cls, **kwargs: [state]),
    )

    model.prepare_att_resources()

    assert model.att_resource_manager.main_gqa_bf16_state is state
    assert not hasattr(model.layers_infer[0], "full_att_state_name")
    assert not hasattr(model.layers_infer[0].indexer, "full_att_state_name")


@pytest.mark.parametrize("role", ["main", "draft0"])
def test_query_selects_role_and_scheme_and_preserves_shared_layers(role: str, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = AttResourceManager(request_capacity=1)
    for model_role in ("main", "draft0"):
        for dtype, groups in ((torch.bfloat16, ((0, 2),)), (torch.float16, ((3,),))):
            manager.set_att_state(model_role, GQAFullAttentionState(
                layout_config=FullAttentionConfig(
                    logical_layer_groups=groups,
                    global_head_num=1,
                    head_dim=128,
                    dtype=dtype,
                ), device="meta",
            ))
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.is_mtp_draft_model = role != "main"
    model.mtp_previous_draft_models = []
    model.max_req_num = 0
    model.torch_memory_saver = SimpleNamespace(region=lambda **_: nullcontext())
    model.main_model = SimpleNamespace(att_resource_manager=manager)
    monkeypatch.setattr("lightllm.common.basemodel.basemodel.AttResourceManager", lambda request_capacity: manager)
    model.config, model.args = {}, SimpleNamespace(mtp_mode=None)
    model.tp_rank_, model.tp_world_size_ = 0, 1
    model.layers_infer = []
    for logical_layer_id in range(4):
        layer = TransformerLayerInfer.__new__(TransformerLayerInfer)
        layer.layer_num_ = logical_layer_id
        model.layers_infer.append(layer)

    model.prepare_att_resources()

    assert all(not hasattr(layer, "full_att_state_name") for layer in model.layers_infer)
    manager.finish_state_creation()
    state = manager.get_att_state(role, "gqa", 0)
    assert state.resource_id == f"{role}_gqa_bf16_state"
    assert manager.get_att_state(role, "gqa", 2) is state
    assert manager.get_att_state(role, "gqa", 3).resource_id == f"{role}_gqa_fp16_state"


@pytest.mark.parametrize("finalized", [False, True])
def test_prefix_and_logical_layer_select_quantized_state(finalized: bool) -> None:
    from dataclasses import replace

    manager = AttResourceManager(request_capacity=1)
    plain = definition("gqa")
    quantized = definition("int8")
    quantized = type(quantized)(
        layout_config=replace(quantized.config, logical_layer_groups=((10, 12),)), device="meta",
    )
    manager.set_att_state("main", plain)
    manager.set_att_state("main", quantized)
    manager.set_att_state("draft0", definition("gqa"))
    logical_layer_id = plain.config.logical_layer_groups[0][0]
    if not finalized:
        with pytest.raises(AssertionError, match="State creation must finish"):
            manager.get_att_state("main", "gqa", logical_layer_id)
    manager.finish_state_creation()

    assert manager.get_att_state("main", "gqa", logical_layer_id) is plain
    assert manager.get_att_state("main", "gqa", 10) is quantized
    assert manager.get_att_state("main", "int8", 10) is quantized
    assert manager.get_att_state("main", "gqa", 12) is quantized
    assert manager.get_att_state("draft0", "gqa", logical_layer_id) is not plain
    with pytest.raises(AssertionError, match="expected exactly one"):
        manager.get_att_state("main", "gqa", 99)


def test_selection_requires_finished_creation_and_reuses_cache() -> None:
    manager = AttResourceManager(request_capacity=1)
    plain = definition("gqa")
    manager.set_att_state("main", plain)
    logical_layer_id = plain.config.logical_layer_groups[0][0]
    with pytest.raises(AssertionError, match="State creation must finish"):
        manager.get_att_state("main", "gqa", logical_layer_id)
    manager.finish_state_creation()
    assert manager.get_att_state("main", "gqa", logical_layer_id) is plain
    hits = manager.get_att_state.cache_info().hits
    assert manager.get_att_state("main", "gqa", logical_layer_id) is plain
    assert manager.get_att_state.cache_info().hits == hits + 1


def test_selection_rejects_ambiguous_finished_members() -> None:
    manager = AttResourceManager(request_capacity=1)
    plain = definition("gqa")
    manager.set_att_state("main", plain)
    manager.set_att_state("main", definition("int8"))
    manager.finish_state_creation()
    with pytest.raises(AssertionError, match="expected exactly one"):
        manager.get_att_state("main", "gqa", plain.config.logical_layer_groups[0][0])


@pytest.mark.parametrize("state_class", [KDAState, GDNState])
def test_bounded_state_layer_ownership(state_class: type[KDAState] | type[GDNState]) -> None:
    state = make_state(cls=state_class)
    manager = AttResourceManager(request_capacity=3)
    manager.set_att_state("main", state)
    manager.finish_state_creation()
    assert state.has_logical_layer(1) and state.has_logical_layer(3)
    assert not state.has_logical_layer(0)
    assert manager.get_att_state("main", state.get_state_type(), 3) is state


def test_mla_selection_excludes_same_layer_tail() -> None:
    from dataclasses import replace

    manager = AttResourceManager(request_capacity=3)
    tail = make_tail()
    template = definition("mla")
    state = type(template)(
        layout_config=replace(template.config, logical_layer_groups=((3, 7),)), device="meta",
    )
    manager.set_att_state("main", state)
    manager.set_att_state("main", tail)
    manager.finish_state_creation()
    assert tail.has_logical_layer(3) and tail.has_logical_layer(7)
    assert not tail.has_logical_layer(0)
    assert manager.get_att_state("main", "mla", 7) is state
    assert manager.get_att_state("main", "kpool", 7) is tail
    assert manager.get_att_state(
        model_name="main",
        att_type="",
        logical_layer_id=7,
        state_class=FullAttentionState,
    ) is state
    with pytest.raises(AssertionError, match="expected exactly one"):
        manager.get_att_state("main", "", 7)



@pytest.mark.parametrize("role", ["draft1", "draft10"])
def test_query_matches_complete_model_role(role: str) -> None:
    manager = AttResourceManager(request_capacity=1)
    for model_name in ("draft1", "draft10"):
        manager.set_att_state(model_name, definition("gqa"))
    manager.finish_state_creation()
    assert manager.get_att_state(role, "gqa", 0).resource_id == f"{role}_gqa_bf16_state"


@pytest.mark.parametrize("algorithm", ["kda", "gdn", "kpool"])
def test_precision_variants_have_distinct_names_and_can_coexist(algorithm: str) -> None:
    from dataclasses import replace

    manager = AttResourceManager(request_capacity=3)
    template = make_tail() if algorithm == "kpool" else make_state(cls=KDAState if algorithm == "kda" else GDNState)
    states = []
    for dtype in (torch.bfloat16, torch.float32):
        config = replace(template.config, **({"dtype": dtype} if algorithm == "kpool" else {"conv_dtype": dtype}))
        state = type(template)(config=config, request_capacity=3, device="cpu")
        manager.set_att_state("main", state)
        states.append(state)
    manager.finish_state_creation()
    logical_layer_id = template.config.logical_layer_ids[0]
    for state in states:
        assert manager.get_att_state("main", state.get_state_type(), logical_layer_id) is state
        assert getattr(manager, state.resource_id) is state
    assert states[0].resource_id != states[1].resource_id
    with pytest.raises(AssertionError, match="expected exactly one"):
        manager.get_att_state("main", algorithm, logical_layer_id)
