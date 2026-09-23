from types import SimpleNamespace

import pytest
import torch

from lightllm.common.eplb_utils import extract_eplb_expert_tensors
from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.models.deepseek_v4.model import (
    DeepseekV4TpPartModel,
    _get_eplb_sampling_peak_nbytes,
    _get_eplb_staging_nbytes,
)
from lightllm.utils import profile_max_tokens


def _expert(rows=4, redundant=2):
    def pack(cols, scale=True):
        return SimpleNamespace(
            weight=torch.empty((rows, cols), dtype=torch.uint8),
            weight_scale=torch.empty((rows, 2), dtype=torch.float32) if scale else None,
        )

    counter = torch.zeros((5, 4), dtype=torch.int64)
    return SimpleNamespace(
        w13=pack(8),
        w2=pack(4, False),
        expert_parallel_state=SimpleNamespace(
            eplb=SimpleNamespace(num_redundant_experts_per_rank=redundant, route_counter=counter)
        ),
    )


@pytest.mark.parametrize("redundant", [0, 2])
def test_full_layout_memory_reserves_one_half_layer_including_scales(redundant):
    weights = [_expert(rows=4 + redundant, redundant=redundant) for _ in range(12)]
    for weight in weights:
        weight.expert_parallel_state.eplb.full_layout = True
        weight.expert_parallel_state.num_primary_experts_per_rank = 4
    expected = sum(
        ((tensor[0].numel() + 1) // 2) * tensor.element_size() * tensor.shape[0]
        for _, tensor in extract_eplb_expert_tensors(weights[0])
    )
    assert _get_eplb_staging_nbytes(weights) == expected


@pytest.mark.parametrize(
    "enable,draft,redundant,staging,sampling,exclusion",
    [
        (True, False, 2, 40, 1536, 40),
        (True, False, 0, 0, 1536, 0),
        (False, False, 2, 0, 0, 0),
        (True, True, 2, 0, 0, 0),
    ],
)
def test_eplb_helpers_and_model_dedupe(enable, draft, redundant, staging, sampling, exclusion):
    expert = _expert(redundant=redundant)
    model = DeepseekV4TpPartModel.__new__(DeepseekV4TpPartModel)
    model.is_mtp_draft_model = draft
    model.args = SimpleNamespace(enable_prefill_eplb=enable)
    model.config = {"num_experts_per_tok": 6}
    model.trans_layers_weight = [SimpleNamespace(experts_=expert), SimpleNamespace(experts_=expert)]
    weights = model._get_eplb_weights()
    assert weights == ([expert] if enable and not draft else [])
    assert _get_eplb_staging_nbytes(weights) == staging
    assert _get_eplb_sampling_peak_nbytes(weights) == sampling
    assert model.get_mtp_profile_weight_exclusion() == exclusion
    assert sum(tensor[0].numel() * tensor.element_size() for _, tensor in extract_eplb_expert_tensors(expert)) == 20
    assert model._get_post_profile_memory_reservations() == {
        name: value for name, value in (("eplb_staging", staging), ("eplb_sampling", sampling)) if value
    }


@pytest.mark.parametrize("exclusion,expected", [(0, 1000), (200, 800), (None, 1000)])
def test_mtp_profile_exclusion_adjustment(monkeypatch, exclusion, expected):
    seen = []
    values = iter((100, 1100))
    monkeypatch.setattr(profile_max_tokens.torch.cuda, "memory_allocated", lambda: next(values))
    monkeypatch.setattr(profile_max_tokens, "get_mtp_weight_layer_num", lambda: 1)
    monkeypatch.setattr(
        profile_max_tokens, "get_mtp_adjusted_mem_fraction", lambda **kw: seen.append(kw["target_weight_bytes"]) or 0.5
    )
    attrs = dict(
        max_total_token_num=None,
        is_mtp_draft_model=False,
        args=SimpleNamespace(mtp_mode="x"),
        config={"n_layer": 1},
        mem_fraction=0.8,
    )
    if exclusion is not None:
        attrs["get_mtp_profile_weight_exclusion"] = lambda: exclusion
    model = SimpleNamespace(**attrs)
    with profile_max_tokens.profile_mtp_weight_memory(model):
        pass
    assert seen == [expected]


@pytest.mark.parametrize("exclusion", [-1, 1001])
def test_mtp_profile_exclusion_validation(monkeypatch, exclusion):
    values = iter((100, 1100))
    monkeypatch.setattr(profile_max_tokens.torch.cuda, "memory_allocated", lambda: next(values))
    model = SimpleNamespace(
        max_total_token_num=None,
        is_mtp_draft_model=False,
        args=SimpleNamespace(mtp_mode="x"),
        config={"n_layer": 1},
        mem_fraction=0.8,
        get_mtp_profile_weight_exclusion=lambda: exclusion,
    )
    with pytest.raises(ValueError, match="invalid MTP profile exclusion"):
        with profile_max_tokens.profile_mtp_weight_memory(model):
            pass


@pytest.mark.parametrize("reservations,expected", [({}, 252), ({"x": 20}, 247)])
def test_memory_manager_profile_reservation_once(monkeypatch, reservations, expected):
    monkeypatch.setattr("lightllm.common.kv_cache_mem_manager.mem_manager.torch.cuda.empty_cache", lambda: None)
    monkeypatch.setattr("lightllm.common.kv_cache_mem_manager.mem_manager.dist.get_world_size", lambda: 1)
    monkeypatch.setattr(
        "lightllm.common.kv_cache_mem_manager.mem_manager.get_available_gpu_memory", lambda w: 1024 / 1024 ** 3
    )
    monkeypatch.setattr("lightllm.common.kv_cache_mem_manager.mem_manager.get_total_gpu_memory", lambda: 0)
    monkeypatch.setattr(MemoryManager, "get_pd_kv_move_buffer_size", lambda self: 0)
    m = MemoryManager.__new__(MemoryManager)
    m.size = None
    m.memory_reservations = reservations
    m.get_cell_size = lambda: 4
    m.get_fixed_memory_size = lambda: 16
    m.profile_size(1)
    assert m.size == expected


def test_full_layout_staging_rounds_odd_rows_and_scalar_scale():
    weight = _expert(rows=3, redundant=0)
    weight.w13.weight = torch.empty((3, 5), dtype=torch.uint8)
    weight.w13.weight_scale = torch.empty((3, 1), dtype=torch.float32)
    weight.w2.weight = torch.empty((3, 3), dtype=torch.uint8)
    weight.expert_parallel_state.eplb.full_layout = True
    weight.expert_parallel_state.num_primary_experts_per_rank = 3
    expected = 3 * ((5 + 1) // 2) + 3 * ((1 + 1) // 2) * 4 + 3 * ((3 + 1) // 2)
    assert _get_eplb_staging_nbytes([weight]) == expected


def test_tile_routing_reservation_is_one_peak_buffer(monkeypatch):
    from lightllm.models.deepseek_v4 import model as dsv4_model
    from lightllm.utils import envs_utils

    monkeypatch.setenv("LIGHTLLM_DSV4_EPLB_TILE_ROUTING", "1")
    envs_utils.get_dsv4_eplb_tile_routing.cache_clear()
    state = SimpleNamespace(eplb=SimpleNamespace(full_layout=True), num_logical_experts=256, world_size=8)
    weights = [SimpleNamespace(expert_parallel_state=state)]
    args = SimpleNamespace(batch_max_tokens=8192, run_mode="prefill")
    assert dsv4_model._get_eplb_tile_routing_peak_nbytes(weights, args, {"num_experts_per_tok": 6}) == 607244
    assert dsv4_model._get_eplb_tile_routing_peak_nbytes(weights * 43, args, {"num_experts_per_tok": 6}) == 607244
    state.eplb.full_layout = False
    assert dsv4_model._get_eplb_tile_routing_peak_nbytes(weights, args, {"num_experts_per_tok": 6}) == 0
    monkeypatch.delenv("LIGHTLLM_DSV4_EPLB_TILE_ROUTING")
    envs_utils.get_dsv4_eplb_tile_routing.cache_clear()
    state.eplb.full_layout = True
    assert dsv4_model._get_eplb_tile_routing_peak_nbytes(weights, args, {"num_experts_per_tok": 6}) == 0
