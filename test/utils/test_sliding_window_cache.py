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


def _memory_manager(big_page_tokens=2048, small_pages=8, enabled=True):
    manager = object.__new__(HybridSlidingMemoryManager)
    manager.head_num, manager.head_dim, manager.layer_num, manager.dtype = 1, 512, 10, torch.bfloat16
    manager.sliding_config = SlidingWindowCacheConfig(
        {i: i for i in range(50)}, {50 + i: i for i in range(10)}, 1024, 4, 256, 1, 512, torch.bfloat16
    )
    manager.big_page_token_num, manager.small_page_num, manager.enable_prompt_cache = (
        big_page_tokens,
        small_pages,
        enabled,
    )
    return manager


@pytest.mark.parametrize("token_num", [1, 2047, 2048, 2049, 8192])
def test_profile_accounts_for_small_big_partial_page_and_hold_token(token_num):
    manager = _memory_manager()
    assert manager.sliding_config.get_state_nbytes() == 200 * 1024 ** 2
    expected = (token_num + 1) * 20480 + (8 + (token_num + 2047) // 2048) * 200 * 1024 ** 2
    assert manager._cache_nbytes(token_num) == expected
    assert manager._profile_token_num(expected) == token_num
    if token_num > 1:
        assert manager._profile_token_num(expected - 1) < token_num


def test_profile_reports_impossible_checkpoint_budget():
    manager = _memory_manager(small_pages=512)
    with pytest.raises(ValueError, match="linear_att_cache_size"):
        manager._profile_token_num(80 * 1024 ** 3)


def test_disabled_prompt_cache_does_not_reserve_pages():
    manager = _memory_manager(small_pages=0, enabled=False)
    assert manager._cache_nbytes(4096) == 4097 * manager.get_cell_size()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_profiled_gpu_pools_match_reserved_bytes_and_are_reused(monkeypatch):
    import lightllm.common.kv_cache_mem_manager.hybrid_sliding_mem_manager as memory_module
    from lightllm.common.req_manager.sliding_window import ReqManagerForSlidingWindow

    manager = _memory_manager(big_page_tokens=32, small_pages=2)
    manager.head_num, manager.head_dim, manager.layer_num = 1, 64, 1
    manager.sliding_config = SlidingWindowCacheConfig({0: 0}, {1: 0}, 32, 1, 64, 1, 64, torch.bfloat16)
    manager.size = None
    budget = manager._cache_nbytes(65)
    monkeypatch.setattr(memory_module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(memory_module, "get_available_gpu_memory", lambda world_size: budget / 1024 ** 3)
    monkeypatch.setattr(memory_module, "get_total_gpu_memory", lambda: 1)
    manager.profile_size(1.0)
    assert manager.size == 65
    manager._init_buffers(manager.size, manager.dtype, manager.head_num, manager.head_dim, manager.layer_num)
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
    manager.size = 1000000
    with pytest.raises(ValueError, match="exceed available GPU memory"):
        manager.profile_size(1.0)


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
