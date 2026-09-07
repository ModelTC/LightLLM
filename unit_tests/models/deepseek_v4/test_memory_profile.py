from types import SimpleNamespace

import pytest
import torch

from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel


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
