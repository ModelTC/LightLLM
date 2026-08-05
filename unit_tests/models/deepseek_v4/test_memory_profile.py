from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager.mem_manager import MemoryManager
from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
from lightllm.utils import profile_max_tokens


@pytest.mark.parametrize(
    "size,draft,expected",
    [(None, False, ["persistent", "manager"]), (7, False, ["manager"]), (None, True, ["manager"])],
)
def test_auto_profile_only_initializes_main_persistent(monkeypatch, size, draft, expected):
    model = DeepseekV4TpPartModel.__new__(DeepseekV4TpPartModel)
    model.max_total_token_num = size
    model.is_mtp_draft_model = draft
    model.args = SimpleNamespace(run_mode="prefill", mtp_step=4, cpu_cache_token_page_size=None)
    model.config = {"n_layer": 1, "head_dim": 512, "index_head_dim": 128, "compress_ratios": [0]}
    model.data_type = torch.bfloat16
    model.mem_fraction = 0.8
    model.max_req_num = 1
    model.req_manager = SimpleNamespace()
    events = []
    monkeypatch.setattr("lightllm.models.deepseek_v4.model.get_added_mtp_kv_layer_num", lambda: 0)
    model._init_auto_profile_persistent_runtime = lambda: events.append("persistent")
    monkeypatch.setattr(
        "lightllm.models.deepseek_v4.model.DeepseekV4MemoryManager",
        lambda *a, **k: events.append("manager") or object(),
    )
    model._init_mem_manager()
    assert events == expected


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
        "lightllm.common.kv_cache_mem_manager.mem_manager.get_available_gpu_memory", lambda w: 1024 / 1024**3
    )
    monkeypatch.setattr("lightllm.common.kv_cache_mem_manager.mem_manager.get_total_gpu_memory", lambda: 0)
    m = MemoryManager.__new__(MemoryManager)
    m.size = None
    m.memory_reservations = reservations
    m.get_cell_size = lambda: 4
    m.get_fixed_memory_size = lambda: 16
    m.profile_size(1)
    assert m.size == expected


def test_rotary_init_is_idempotent_and_late_binds(monkeypatch):
    model = DeepseekV4TpPartModel.__new__(DeepseekV4TpPartModel)
    model.config = {
        "qk_rope_head_dim": 4,
        "rope_theta": 10000,
        "compress_rope_theta": 100000,
        "max_position_embeddings": 16,
    }
    model.max_seq_length = 8
    torch_arange = torch.arange

    def cpu_arange(*args, **kwargs):
        kwargs["device"] = "cpu"
        return torch_arange(*args, **kwargs)

    monkeypatch.setattr("lightllm.models.deepseek_v4.model.torch.arange", cpu_arange)
    model._init_to_get_rotary()
    sliding_freqs = model._freqs_cis_sliding
    compress_freqs = model._freqs_cis_compress
    model.layers_infer = [
        SimpleNamespace(compress_ratio=0, index_infer=SimpleNamespace()),
        SimpleNamespace(compress_ratio=4, index_infer=SimpleNamespace()),
    ]
    model._init_to_get_rotary()
    assert model._freqs_cis_sliding is sliding_freqs
    assert model._freqs_cis_compress is compress_freqs
    assert model.layers_infer[0].freqs_cis is sliding_freqs
    assert model.layers_infer[1].freqs_cis is compress_freqs
    assert model.layers_infer[0].index_infer.freqs_cis is compress_freqs
    assert model.layers_infer[1].index_infer.freqs_cis is compress_freqs
