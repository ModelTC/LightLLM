from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager.hybrid_sliding_mem_manager import HybridSlidingMemoryManager
from lightllm.common.req_manager.linear_att import ReqManagerForMamba
from lightllm.common.sliding_window_cache_manager import SlidingWindowCacheConfig
from lightllm.models.gemma4.kv_layout import get_kv_cache_layout


@pytest.mark.parametrize("layer_num,shared,sliding_num,full_num", [(42, 18, 20, 4), (60, 0, 50, 10)])
def test_gemma_physical_owners_and_last_readers(layer_num, shared, sliding_num, full_num):
    layer_types = ["sliding_attention"] * 5 + ["full_attention"]
    maps, owners, last_readers = get_kv_cache_layout(
        {"layer_types": layer_types * (layer_num // 6), "num_kv_shared_layers": shared}
    )
    assert len(set(maps["sliding_attention"].values())) == sliding_num
    assert len(set(maps["full_attention"].values())) == full_num
    for index, owner in enumerate(owners):
        assert owner <= index <= last_readers[owner]
    if shared:
        assert owners[40] == 22 and last_readers[22] == 40
        assert owners[41] == 23 and last_readers[23] == 41


def _memory_manager(big_page_tokens=2048, small_pages=8, enabled=True, cpu_cache=False):
    manager = object.__new__(HybridSlidingMemoryManager)
    manager.size = None
    manager.head_num, manager.head_dim, manager.layer_num, manager.dtype = 1, 512, 10, torch.bfloat16
    manager.sliding_config = SlidingWindowCacheConfig(
        {i: i for i in range(50)}, {50 + i: i for i in range(10)}, 1024, 4, 256, 1, 512, torch.bfloat16
    )
    manager.big_page_token_num, manager.small_page_num, manager.enable_prompt_cache = (
        big_page_tokens,
        small_pages,
        enabled,
    )
    manager.cpu_cache_temp_page_num = 2 if cpu_cache else 0
    return manager


def _required_bytes(manager, token_num):
    big_pages = (token_num + manager.big_page_token_num - 1) // manager.big_page_token_num
    state_pages = manager.small_page_num + manager.cpu_cache_temp_page_num
    if manager.enable_prompt_cache:
        state_pages += big_pages
    return (token_num + 1) * manager.get_cell_size() + state_pages * manager.sliding_config.get_state_nbytes()


def _profile_with_budget(monkeypatch, manager, available_bytes, mem_fraction=1.0, total_bytes=16 * 1024 ** 3):
    import lightllm.common.kv_cache_mem_manager.hybrid_sliding_mem_manager as memory_module

    monkeypatch.setattr(memory_module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(memory_module, "get_available_gpu_memory", lambda world_size: available_bytes / 1024 ** 3)
    monkeypatch.setattr(memory_module, "get_total_gpu_memory", lambda: total_bytes / 1024 ** 3)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    manager.profile_size(mem_fraction)


@pytest.mark.parametrize("token_num", [1, 2047, 2048, 2049, 8192])
def test_profile_accounts_for_small_big_partial_page_and_hold_token(monkeypatch, token_num):
    manager = _memory_manager()
    assert manager.sliding_config.get_state_nbytes() == 200 * 1024 ** 2
    expected = (token_num + 1) * 20480 + (8 + (token_num + 2047) // 2048) * 200 * 1024 ** 2
    assert _required_bytes(manager, token_num) == expected
    _profile_with_budget(monkeypatch, manager, expected)
    assert manager.size == token_num
    manager.size = None
    if token_num > 1:
        _profile_with_budget(monkeypatch, manager, expected - 1)
        assert manager.size == token_num - 1
    else:
        with pytest.raises(ValueError, match="Insufficient GPU memory"):
            _profile_with_budget(monkeypatch, manager, expected - 1)


@pytest.mark.parametrize("leftover", [1, 200 * 1024 ** 2 - 1, 200 * 1024 ** 2])
def test_profile_cannot_start_next_page_without_checkpoint_and_token_budget(monkeypatch, leftover):
    manager = _memory_manager()
    _profile_with_budget(monkeypatch, manager, _required_bytes(manager, 2048) + leftover)
    assert manager.size == 2048


@pytest.mark.parametrize("available_bytes", [-1, 0, 80 * 1024 ** 3])
def test_profile_reports_impossible_checkpoint_budget(monkeypatch, available_bytes):
    manager = _memory_manager(small_pages=512)
    with pytest.raises(ValueError, match="linear_att_cache_size"):
        _profile_with_budget(monkeypatch, manager, available_bytes)


def test_disabled_prompt_cache_does_not_reserve_pages(monkeypatch):
    manager = _memory_manager(small_pages=0, enabled=False)
    _profile_with_budget(monkeypatch, manager, 4097 * manager.get_cell_size())
    assert manager.size == 4096


def test_cpu_cache_reserves_two_additional_window_checkpoints(monkeypatch):
    gpu_only = _memory_manager()
    cpu_cache = _memory_manager(cpu_cache=True)
    gpu_budget = _required_bytes(gpu_only, 4096)
    cpu_budget = gpu_budget + 2 * cpu_cache.sliding_config.get_state_nbytes()
    _profile_with_budget(monkeypatch, gpu_only, gpu_budget)
    _profile_with_budget(monkeypatch, cpu_cache, cpu_budget)
    assert gpu_only.size == cpu_cache.size == 4096


def test_explicit_size_is_checked_against_complete_cache_budget(monkeypatch):
    manager = _memory_manager(cpu_cache=True)
    manager.size = 2049
    budget = _required_bytes(manager, manager.size)
    _profile_with_budget(monkeypatch, manager, budget)
    assert manager.size == 2049
    with pytest.raises(ValueError, match="exceed available GPU memory"):
        _profile_with_budget(monkeypatch, manager, budget - 1)


@pytest.mark.parametrize("explicit_size", [None, 2049])
def test_mem_fraction_reserves_headroom_only_for_automatic_size(monkeypatch, explicit_size):
    manager = _memory_manager()
    manager.size = explicit_size
    total_bytes, mem_fraction = 8 * 1024 ** 3, 0.5
    budget = _required_bytes(manager, 2048) + total_bytes // 2
    _profile_with_budget(monkeypatch, manager, budget, mem_fraction=mem_fraction, total_bytes=total_bytes)
    assert manager.size == (2048 if explicit_size is None else explicit_size)


@pytest.mark.parametrize("disabled", [False, True])
def test_page_pools_follow_active_prompt_cache_flag(monkeypatch, disabled):
    import lightllm.common.kv_cache_mem_manager.hybrid_sliding_mem_manager as memory_module

    args = SimpleNamespace(
        use_dynamic_prompt_cache=False,
        disable_dynamic_prompt_cache=disabled,
        enable_cpu_cache=False,
        linear_att_cache_size=3,
        linear_att_hash_page_size=32,
        linear_att_page_block_num=8,
    )
    monkeypatch.setattr(memory_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(memory_module.MemoryManager, "__init__", lambda self, **kwargs: None)
    config = SlidingWindowCacheConfig({0: 0}, {1: 0}, 32, 1, 64, 1, 64, torch.bfloat16)
    manager = HybridSlidingMemoryManager(size=256, sliding_config=config)
    assert manager.enable_prompt_cache is not disabled
    assert manager.small_page_num == (0 if disabled else 3)
    assert manager.big_page_token_num == 256
    assert manager.cpu_cache_temp_page_num == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("enabled,cpu_cache", [(False, False), (True, False), (True, True)])
def test_profiled_gpu_pools_match_reserved_bytes_and_are_reused(monkeypatch, enabled, cpu_cache):
    from lightllm.common.req_manager.sliding_window import ReqManagerForSlidingWindow

    manager = _memory_manager(big_page_tokens=32, small_pages=2 if enabled else 0, enabled=enabled, cpu_cache=cpu_cache)
    manager.head_num, manager.head_dim, manager.layer_num = 1, 64, 1
    manager.sliding_config = SlidingWindowCacheConfig({0: 0}, {1: 0}, 32, 1, 64, 1, 64, torch.bfloat16)
    budget = _required_bytes(manager, 65)
    _profile_with_budget(monkeypatch, manager, budget)
    assert manager.size == 65
    manager._init_buffers(manager.size, manager.dtype, manager.head_num, manager.head_dim, manager.layer_num)
    assert manager.linear_att_big_page_buffers.size == (3 if enabled else 0) + (2 if cpu_cache else 0)
    assert manager.linear_att_big_page_buffers.get_free_cache_num() == (3 if enabled else 0)
    assert manager.sliding_small_page_buffers.size == (2 if enabled else 0)
    allocated = sum(
        t.numel() * t.element_size()
        for t in [
            manager.kv_buffer,
            manager.linear_att_big_page_buffers.state_cache,
            manager.sliding_small_page_buffers.state_cache,
        ]
    )
    assert allocated == budget
    req_manager = object.__new__(ReqManagerForSlidingWindow)
    req_manager.mem_manager = manager
    assert req_manager.create_state_cache_manager(2) is manager.sliding_small_page_buffers


@pytest.mark.parametrize("mtp_step", [0, 2])
def test_linear_small_page_preserves_main_copy_and_mtp_crop(mtp_step):
    manager = object.__new__(ReqManagerForMamba)
    manager.mtp_step = mtp_step
    manager.linear_config = SimpleNamespace(get_conv_state_shape=lambda: (3, 4))
    conv = torch.arange(2 * 3 * 3 * (4 + mtp_step)).reshape(2, 3, 3, 4 + mtp_step)
    ssm = torch.arange(2 * 3 * (mtp_step + 1) * 5).reshape(2, 3 * (mtp_step + 1), 5)
    manager.req_to_conv_state, manager.req_to_ssm_state = SimpleNamespace(buffer=conv), SimpleNamespace(buffer=ssm)
    dst_conv, dst_ssm = torch.empty((2, 3, 4), dtype=conv.dtype), torch.empty((2, 5), dtype=ssm.dtype)
    pages = SimpleNamespace(get_state_cache=lambda buffer_idx: (dst_conv, dst_ssm))
    manager.save_small_page_state(1, 0, pages)
    torch.testing.assert_close(dst_conv, conv[:, 1, :, :4])
    torch.testing.assert_close(dst_ssm, ssm[:, mtp_step + 1])


@pytest.mark.parametrize("unsupported_mode", ["enable_dp_prompt_cache_fetch", "diverse_mode"])
def test_unsupported_sliding_state_transfer_modes_fail_before_loading_weights(monkeypatch, unsupported_mode):
    import lightllm.models.gemma4.model as gemma_model

    model = object.__new__(gemma_model.Gemma4TpPartModel)
    model.load_way, model.tp_world_size_ = "HF", 2
    model.config = {"num_attention_heads": 8, "num_key_value_heads": 2, "num_hidden_layers": 42}
    args = SimpleNamespace(
        mtp_step=0,
        enable_cpu_cache=False,
        disable_chunked_prefill=False,
        run_mode="normal",
        llm_kv_type="None",
        enable_dp_prompt_cache_fetch=False,
        diverse_mode=False,
    )
    setattr(args, unsupported_mode, True)
    monkeypatch.setattr(gemma_model, "get_env_start_args", lambda: args)
    with pytest.raises(AssertionError, match="does not support"):
        model._verify_params()


@pytest.mark.parametrize("shared_layers", [0, 18])
@pytest.mark.parametrize(
    "overlap_mode", [None, "enable_prefill_microbatch_overlap", "enable_decode_microbatch_overlap"]
)
def test_shared_kv_rejects_interleaved_microbatches(monkeypatch, shared_layers, overlap_mode):
    import lightllm.models.gemma4.model as gemma_model

    model = object.__new__(gemma_model.Gemma4TpPartModel)
    model.load_way, model.tp_world_size_ = "HF", 2
    model.config = {
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "num_hidden_layers": 42,
        "num_kv_shared_layers": shared_layers,
    }
    args = SimpleNamespace(
        mtp_step=0,
        enable_cpu_cache=False,
        disable_chunked_prefill=False,
        run_mode="normal",
        llm_kv_type="None",
        enable_dp_prompt_cache_fetch=False,
        diverse_mode=False,
        enable_prefill_microbatch_overlap=False,
        enable_decode_microbatch_overlap=False,
    )
    if overlap_mode is not None:
        setattr(args, overlap_mode, True)
    monkeypatch.setattr(gemma_model, "get_env_start_args", lambda: args)
    if shared_layers and overlap_mode is not None:
        with pytest.raises(AssertionError, match="shared sliding-window KV does not support microbatch overlap"):
            model._verify_params()
    else:
        model._verify_params()
