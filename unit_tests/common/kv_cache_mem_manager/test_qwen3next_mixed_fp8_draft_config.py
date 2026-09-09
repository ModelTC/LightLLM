from types import SimpleNamespace

import torch

from lightllm.common.linear_att_cache_manager import config_objs
from lightllm.common.linear_att_cache_manager.config_objs import LinearAttCacheConfig
from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextMemManager


def _config(draft_dtype):
    return LinearAttCacheConfig(
        tp_world_size=2,
        full_att_all_num_kv_heads=4,
        full_att_dtype=torch.uint8,
        full_att_num_kv_heads=2,
        full_att_head_dim=256,
        global_linear_k_heads=16,
        global_linear_v_heads=48,
        num_linear_k_heads=8,
        num_linear_v_heads=24,
        head_linear_k_dim=128,
        head_linear_v_dim=128,
        conv_kernel_size=4,
        linear_layer_num=48,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.bfloat16,
        full_attention_interval=4,
        all_layer_num=64,
        draft_full_att_kv_layer_num=3,
        draft_full_att_dtype=draft_dtype,
    )


def test_mixed_fp8_target_bf16_draft_cpu_page_segments(monkeypatch):
    monkeypatch.setattr(
        config_objs,
        "get_env_start_args",
        lambda: SimpleNamespace(linear_att_page_block_num=8, linear_att_hash_page_size=2048, cpu_cache_token_page_size=16384),
    )
    config = _config(torch.bfloat16)
    assert config.use_mixed_target_fp8_draft_bf16()
    assert config.get_target_full_att_kv_layer_num() == 16
    assert config.get_full_att_kv_layer_num_with_draft_model() == 19
    assert config.get_cpu_cache_target_full_att_bytes() == 536870912
    assert config.get_cpu_cache_draft_full_att_bytes() == 201326592
    assert config.get_cpu_cache_full_att_bytes() == 738197504


def test_mixed_cell_size_and_user_cpu_page_size(monkeypatch):
    monkeypatch.setattr(
        config_objs,
        "get_env_start_args",
        lambda: SimpleNamespace(linear_att_page_block_num=8, linear_att_hash_page_size=1024, cpu_cache_token_page_size=8192),
    )
    config = _config(torch.bfloat16)
    assert config.get_cpu_cache_target_full_att_bytes() == 268435456
    assert config.get_cpu_cache_draft_full_att_bytes() == 100663296
    manager = object.__new__(Qwen3NextMemManager)
    manager.linear_config = config
    manager.target_full_att_layer_num = 16
    manager.head_num = 2
    manager.head_dim = 256
    manager.dtype = torch.uint8
    assert manager.get_cell_size() == 2 * 2 * 256 * 16 + 2 * 2 * 256 * 3 * 2


def test_legacy_fp8_layout_keeps_single_dtype(monkeypatch):
    monkeypatch.setattr(
        config_objs,
        "get_env_start_args",
        lambda: SimpleNamespace(linear_att_page_block_num=8, linear_att_hash_page_size=2048, cpu_cache_token_page_size=16384),
    )
    config = _config(None)
    assert not config.use_mixed_target_fp8_draft_bf16()
    assert config.get_draft_full_att_dtype() is torch.uint8
    assert config.get_cpu_cache_full_att_bytes() == 637534208


def test_bf16_legacy_layout_remains_single_segment(monkeypatch):
    monkeypatch.setattr(
        config_objs,
        "get_env_start_args",
        lambda: SimpleNamespace(linear_att_page_block_num=8, linear_att_hash_page_size=1024, cpu_cache_token_page_size=8192),
    )
    config = _config(None)
    config.full_att_dtype = torch.bfloat16
    assert not config.use_mixed_target_fp8_draft_bf16()
    assert config.get_draft_full_att_dtype() is torch.bfloat16
    # Legacy BF16 retains its one physical KV buffer, including the draft
    # layers; no mixed-buffer path is selected.
    assert config.get_cpu_cache_draft_full_att_bytes() == 100663296
    assert config.get_cpu_cache_full_att_bytes() == 637534208


def test_mixed_snapshot_restore_writes_both_buffers_for_tensor_indices():
    manager = object.__new__(Qwen3NextMemManager)
    manager.kv_buffer = torch.arange(2 * 6 * 2 * 2, dtype=torch.uint8).view(2, 6, 2, 2)
    manager.draft_kv_buffer = torch.arange(3 * 6 * 2 * 2, dtype=torch.bfloat16).view(3, 6, 2, 2)
    indices = torch.tensor([1, 4], dtype=torch.long)
    snapshot = manager.get_index_kv_buffer(indices)

    manager.kv_buffer.zero_()
    manager.draft_kv_buffer.zero_()
    manager.load_index_kv_buffer(indices, snapshot)

    assert torch.equal(manager.kv_buffer[:, indices], snapshot["kv_buffer"])
    assert torch.equal(manager.draft_kv_buffer[:, indices], snapshot["draft_kv_buffer"])
    assert not torch.any(manager.kv_buffer[:, [0, 2, 3, 5]])
    assert not torch.any(manager.draft_kv_buffer[:, [0, 2, 3, 5]])
