import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager.qwen3next_mem_manager import Qwen3NextMemManager
from lightllm.common.req_manager.linear_att import ReqManagerForMamba
from lightllm.common.state_cache_manager import LinearAttCacheConfig
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.envs_utils import get_env_start_args, get_unique_server_name
from lightllm.utils.kv_cache_utils import CpuKVCacheMeta


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(params=[False, True], ids=["default_chunks", "whole_mm_blocks"])
def linear_cache(monkeypatch, tmp_path, request):
    args = StartArgs(
        tp=1,
        page_size=4,
        max_req_total_len=512,
        linear_att_hash_page_size=64,
        linear_att_page_block_num=2,
        cpu_cache_token_page_size=128,
        disable_chunked_mm_input=request.param,
    )
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(dataclasses.asdict(args)))
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_NODE", "0")
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_DP", "0")
    monkeypatch.setenv("LIGHTLLM_DP_WORLD_SIZE", "1")
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", tmp_path.name)
    monkeypatch.setattr("lightllm.common.req_manager.req_sampling_params.get_vocab_size", lambda _: 128)
    get_env_start_args.cache_clear()
    get_unique_server_name.cache_clear()
    config = LinearAttCacheConfig(
        tp_world_size=1,
        full_att_all_num_kv_heads=1,
        full_att_dtype=torch.bfloat16,
        full_att_num_kv_heads=1,
        full_att_head_dim=8,
        global_linear_k_heads=1,
        global_linear_v_heads=1,
        num_linear_k_heads=1,
        num_linear_v_heads=1,
        head_linear_k_dim=8,
        head_linear_v_dim=8,
        conv_kernel_size=3,
        linear_layer_num=1,
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
        full_attention_interval=2,
        all_layer_num=2,
    )
    mem = Qwen3NextMemManager(384, torch.bfloat16, 1, 8, 1, config)
    requests = ReqManagerForMamba(2, 512, mem, config)
    requests.create_small_page_cache_manager(2)
    monkeypatch.setattr(g_infer_context, "is_hybrid_att_model", True)
    monkeypatch.setattr(g_infer_context, "req_manager", requests)
    monkeypatch.setattr(
        g_infer_context,
        "radix_cache",
        SimpleNamespace(
            get_big_page_ids_by_node=lambda node: [],
            free_one_small_page_buffer=lambda: None,
            small_page_buffers=requests.small_page_buffers,
        ),
    )
    yield mem, requests, config
    get_env_start_args.cache_clear()
    get_unique_server_name.cache_clear()


@pytest.mark.parametrize("token_num", [320, 384])
def test_cpu_roundtrip_keeps_kv_continuous_and_restores_only_saved_states(
    monkeypatch, linear_cache, cpu_cache_factory, token_num
):
    from lightllm.server.router.dynamic_prompt.hybrid_att_radix_cache import HybridAttPagedRadixCache
    from lightllm.server.router.model_infer.mode_backend import multi_level_kv_cache as common
    from lightllm.server.router.model_infer.mode_backend.multi_level_kv_cache import MultiLevelKvCacheModule

    mem, requests, config = linear_cache
    disable_chunked_mm_input = get_env_start_args().disable_chunked_mm_input
    first_page_has_state = not disable_chunked_mm_input
    big = mem.big_page_buffers
    big_states = {}
    for end in range(128, token_num + 1, 128):
        slot = big.alloc_one_state_cache()
        big_states[end] = slot
        for tensor in big.get_state_cache(slot):
            tensor.normal_()
    tail_slot = None
    if token_num % 128:
        tail_slot = requests.small_page_buffers.alloc_one_state_cache()
        for tensor in requests.small_page_buffers.get_state_cache(tail_slot):
            tensor.normal_()
        final_states = requests.small_page_buffers.get_state_cache(tail_slot)
    else:
        final_states = big.get_state_cache(big_states[token_num])
    expected_states = [tensor.clone() for tensor in final_states]
    expected_big_states = {end: [t.clone() for t in big.get_state_cache(slot)] for end, slot in big_states.items()}
    mem.kv_buffer.normal_()
    expected_kv = mem.kv_buffer[:, :token_num].clone()
    client = cpu_cache_factory(
        CpuKVCacheMeta(3, 1, 1, 1, config.get_cpu_cache_big_page_bytes(), torch.uint8, 0, torch.uint8),
    )
    pages, _ = client.allocate_pages([11, 22, 33], disk_offload_enable=False)
    indexes = torch.arange(token_num, dtype=torch.int32, device="cuda")
    page_ids = torch.arange(3, dtype=torch.int32, device="cuda")
    req = SimpleNamespace(
        req_idx=0,
        cur_kv_len=token_num,
        hold_kv_len=token_num,
        hybrid_cache_len=token_num,
        hybrid_len_to_big_page_id=big_states,
        tail_small_page_buffer_id=tail_slot,
        shared_kv_node=None,
        mm_input_spans=[(96, 160)] if disable_chunked_mm_input else [],
        sampling_param=SimpleNamespace(shm_param=SimpleNamespace(prompt_logprobs=-1)),
        shm_req=SimpleNamespace(
            input_len=token_num + 1,
            disk_prompt_cache_len=0,
            cpu_cache_match_page_indexes=SimpleNamespace(get_all=lambda: pages),
            token_hash_page_len_list=SimpleNamespace(get_all=lambda: [128, 256, token_num]),
        ),
    )

    mem.operator.offload_gpu_kv_to_cpu_cache(indexes, page_ids, torch.zeros_like(page_ids), client, req)
    torch.cuda.synchronize()
    module = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    module.cpu_cache_client = client
    client.update_pages_status_to_ready(pages, deref=False)
    big.free_state_cache(list(big_states.values()))
    if tail_slot is not None:
        requests.small_page_buffers.free_state_cache([tail_slot])
    req.hybrid_len_to_big_page_id = {}
    req.tail_small_page_buffer_id = None
    req.cur_kv_len = req.hold_kv_len = 0
    mem.kv_buffer.zero_()
    big.conv_state_cache.buffer.zero_()
    big.ssm_state_cache.buffer.zero_()

    def allocate(req, count):
        allocated = mem.alloc(count)
        requests.req_to_token_indexs[req.req_idx, :count] = allocated.cuda()
        req.hold_kv_len = count
        return allocated

    module.args = get_env_start_args()
    module.backend = SimpleNamespace(
        is_master_in_dp=True,
        _alloc_req_kv_mem=allocate,
        model=SimpleNamespace(mem_manager=mem, req_manager=requests),
    )
    module.init_sync_group = None
    module.need_sync_compute_stream = lambda: False
    monkeypatch.setattr(common.g_infer_context, "get_can_alloc_token_num", lambda: 384)
    monkeypatch.setattr(common.dist, "barrier", lambda **kwargs: None)
    module.load_cpu_cache_to_reqs([req])

    assert req.cur_kv_len == req.hold_kv_len == token_num
    assert torch.equal(mem.kv_buffer[:, :token_num], expected_kv)
    assert sorted(req.hybrid_len_to_big_page_id) == list(range(128, token_num + 1, 128))
    # Even the invalid page is copied normally; only restoring from it is forbidden.
    for end, slot in req.hybrid_len_to_big_page_id.items():
        for actual, expected in zip(big.get_state_cache(slot), expected_big_states[end]):
            torch.testing.assert_close(actual, expected)
    restored_state = requests.linear_state_manager.get_accepted_state(req.req_idx)
    torch.testing.assert_close(restored_state.conv.cpu(), expected_states[0])
    torch.testing.assert_close(restored_state.ssm.cpu(), expected_states[1])

    # Loaded intermediate pages must remain unusable as radix restore points.
    tree = HybridAttPagedRadixCache(384, 103, 64, 2, mem, requests.small_page_buffers)
    cache_len = max(req.hybrid_len_to_big_page_id)
    tree.insert(
        torch.arange(cache_len),
        block_hashs=list(range(1, cache_len // 64 + 1)),
        block_state_idxs=[None] * (cache_len // 64),
        len_to_big_page_id=req.hybrid_len_to_big_page_id,
    )
    for length, expected in [(128, 128 if first_page_has_state else 0), (256, 256)]:
        node, matched, _ = tree.match_prefix(
            torch.arange(length),
            block_hashs=list(range(1, length // 64 + 1)),
            update_refs=True,
            mm_input_spans=req.mm_input_spans,
        )
        assert matched == expected
        tree.dec_node_ref_counter(node)
    req.shared_kv_node, _, _ = tree.match_prefix(
        torch.arange(cache_len),
        block_hashs=list(range(1, cache_len // 64 + 1)),
        update_refs=True,
        mm_input_spans=req.mm_input_spans,
    )
    tree.dec_node_ref_counter(req.shared_kv_node)
