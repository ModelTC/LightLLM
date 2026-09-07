from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lightllm.common.sliding_window_cache_manager import SlidingWindowCacheConfig
from lightllm.models.gemma4.kv_layout import build_sliding_cache_config


def _gemma_config(shared):
    layer_num = 42 if shared else 60
    return {
        "model_type": "gemma4_text",
        "num_hidden_layers": layer_num,
        "num_attention_heads": 8,
        "num_key_value_heads": 2 if shared else 16,
        "num_global_key_value_heads": None if shared else 4,
        "head_dim": 256,
        "global_head_dim": 512,
        "sliding_window": 512 if shared else 1024,
        "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * (layer_num // 6),
        "num_kv_shared_layers": 18 if shared else 0,
    }


@pytest.mark.parametrize("shared,tp_world_size", [(True, 1), (True, 2), (False, 1), (False, 2), (False, 4)])
def test_cpu_page_layout_uses_physical_owners_and_all_tp_shards(shared, tp_world_size):
    config = _gemma_config(shared)
    layout = build_sliding_cache_config(config, tp_world_size, torch.bfloat16)
    full_layers, sliding_layers = (4, 20) if shared else (10, 50)
    full_heads = config["num_global_key_value_heads"] or config["num_key_value_heads"]
    big_page_tokens = 2048
    full_bytes = full_layers * big_page_tokens * 2 * full_heads * config["global_head_dim"] * 2
    state_bytes = sliding_layers * config["sliding_window"] * 2 * config["num_key_value_heads"] * config["head_dim"] * 2
    assert layout.get_cpu_cache_full_att_bytes(big_page_tokens, tp_world_size) == full_bytes
    assert layout.get_cpu_cache_state_bytes(tp_world_size) == state_bytes
    assert layout.get_cpu_cache_big_page_bytes(big_page_tokens, tp_world_size) == full_bytes + state_bytes


def test_cpu_page_payload_is_aligned_without_changing_section_sizes():
    layout = SlidingWindowCacheConfig({0: 0}, {1: 0}, 1, 1, 3, 1, 5, torch.bfloat16)
    assert layout.get_cpu_cache_full_att_bytes(3, 1) == 60
    assert layout.get_cpu_cache_state_bytes(1) == 12
    assert layout.get_cpu_cache_big_page_bytes(3, 1) == 80


@pytest.mark.parametrize("shared,tp_world_size", [(True, 4), (False, 8)])
def test_cpu_layout_rejects_replicated_kv_heads(shared, tp_world_size):
    with pytest.raises(AssertionError, match="KV heads must be divisible"):
        build_sliding_cache_config(_gemma_config(shared), tp_world_size, torch.bfloat16)


@pytest.mark.parametrize("wrapped", [False, True])
def test_sliding_cpu_meta_is_a_flat_global_payload(monkeypatch, wrapped):
    import lightllm.utils.kv_cache_utils as cache_utils

    config = _gemma_config(True)
    layout = build_sliding_cache_config(config, 2, torch.bfloat16)
    page_bytes = layout.get_cpu_cache_big_page_bytes(2048, 2)
    args = SimpleNamespace(
        model_dir="gemma-test",
        enable_cpu_cache=True,
        tp=4,
        dp=2,
        linear_att_hash_page_size=128,
        linear_att_page_block_num=16,
        cpu_cache_token_page_size=2048,
        cpu_cache_storage_size=3 * page_bytes / 1024 ** 3,
        mtp_mode=None,
    )
    monkeypatch.setattr(cache_utils, "get_env_start_args", lambda: args)
    monkeypatch.setattr(cache_utils, "is_linear_att_mixed_model", lambda _: False)
    monkeypatch.setattr(cache_utils, "is_sliding_att_mixed_model", lambda _: True)
    monkeypatch.setattr(cache_utils, "get_llm_data_type", lambda: torch.bfloat16)
    monkeypatch.setattr(cache_utils, "get_config_json", lambda _: {"text_config": config} if wrapped else config)
    meta = cache_utils.calcu_cpu_cache_meta.__wrapped__()
    assert meta.data_type == torch.uint8
    assert (meta.layer_num, meta.token_page_size, meta.num_heads) == (1, 1, 1)
    assert meta.head_dim == meta.calcu_one_page_size() == page_bytes
    assert meta.page_num == 3
    assert args.cpu_cache_token_page_size == 2048


def test_hybrid_request_initializes_cpu_hashes_without_linear_state(monkeypatch):
    import lightllm.server.core.objs.req as req_module

    prompt = list(range(18))
    args = SimpleNamespace(
        model_dir="gemma-test",
        mtp_step=0,
        enable_cpu_cache=True,
        linear_att_hash_page_size=4,
        linear_att_page_block_num=3,
        cpu_cache_token_page_size=12,
    )
    monkeypatch.setattr(req_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(req_module, "is_hybrid_att_mixed_model", lambda _: True)
    req = SimpleNamespace(index_in_shm_mem=0, ref_count=0)
    req.create_logprobs_shm_array = lambda: None
    req.create_prompt_ids_shm_array = lambda: setattr(
        req, "shm_prompt_ids", SimpleNamespace(arr=np.empty(2048, dtype=np.int64))
    )
    req.post_init = lambda: None
    req.get_prompt_ids = lambda: prompt
    req._fill_linear_att_token_hash = lambda: req_module.Req._fill_linear_att_token_hash(req)
    req._calcu_linear_att_cpu_cache_page_len_list = lambda: req_module.Req._calcu_linear_att_cpu_cache_page_len_list(
        req
    )
    req_module.Req.init(req, 0, prompt, req_module.SamplingParams(), tokenizer=None, chunked_prefill_size=16)
    hashes = req.linear_att_token_hash_list.get_all()
    assert len(hashes) == 4
    assert req.token_hash_list.get_all() == [hashes[2], hashes[3]]
    assert req.token_hash_page_len_list.get_all() == [12, 16]
    assert req.cpu_cache_match_page_indexes.get_all() == []


@pytest.mark.parametrize("disable_tail,tail_buffer", [(False, None), (False, 3), (True, 3)])
def test_sliding_offload_uses_existing_hybrid_tail_policy(monkeypatch, disable_tail, tail_buffer):
    import lightllm.server.router.model_infer.mode_backend.multi_level_kv_cache as cache_module

    module = object.__new__(cache_module.MultiLevelKvCacheModule)
    module.args = SimpleNamespace(cpu_cache_token_page_size=16, disable_linear_att_small_page_cpu_cache=disable_tail)
    monkeypatch.setattr(cache_module.g_infer_context, "is_linear_att_mixed_model", False)
    monkeypatch.setattr(cache_module.g_infer_context, "is_hybrid_att_mixed_model", True)
    req = SimpleNamespace(tail_linear_att_small_page_buffer_id=tail_buffer)
    expected_pages = 2 if not disable_tail and tail_buffer is not None else 1
    assert module._handle_linear_att_last_page(req, 2, [16, 20]) == expected_pages


@pytest.mark.parametrize("enable_cpu_cache,disable_gpu_cache", [(True, False), (True, True), (False, True)])
def test_cpu_state_transfer_requires_gpu_prefix_cache(monkeypatch, enable_cpu_cache, disable_gpu_cache):
    import lightllm.models.gemma4.model as gemma_model

    model = object.__new__(gemma_model.Gemma4TpPartModel)
    model.load_way, model.tp_world_size_ = "HF", 2
    model.config = _gemma_config(True)
    args = SimpleNamespace(
        mtp_step=0,
        enable_cpu_cache=enable_cpu_cache,
        disable_dynamic_prompt_cache=disable_gpu_cache,
        disable_chunked_prefill=False,
        run_mode="normal",
        llm_kv_type="None",
        enable_dp_prompt_cache_fetch=False,
        diverse_mode=False,
        enable_prefill_microbatch_overlap=False,
        enable_decode_microbatch_overlap=False,
    )
    monkeypatch.setattr(gemma_model, "get_env_start_args", lambda: args)
    if enable_cpu_cache and disable_gpu_cache:
        with pytest.raises(AssertionError, match="CPU cache requires GPU prefix cache"):
            model._verify_params()
    else:
        model._verify_params()
