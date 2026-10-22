import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.kv_cache_mem_manager import MemoryManager
from lightllm.common.req_manager import ReqManager
from lightllm.common.cache_coordinator import HybridCacheCoordinator
from lightllm.common.linear_state import LinearStateManager
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
    monkeypatch.setattr("lightllm.common.kv_cache_mem_manager.mem_manager.get_num_key_value_heads", lambda _: 1)
    mem = MemoryManager(384, torch.bfloat16, 1, 8, 1)
    mem.set_kv_layer_mapping(config.get_full_att_kv_layer_mapping())
    requests = ReqManager(2, 512, mem)
    coordinator = HybridCacheCoordinator(LinearStateManager(2, config, args))
    coordinator.bind_mem_manager(mem)
    coordinator.create_small_page_cache_manager(2)
    monkeypatch.setattr(g_infer_context, "cache_coordinator", None)
    monkeypatch.setattr(g_infer_context, "is_hybrid_att_model", True)
    monkeypatch.setattr(g_infer_context, "req_manager", requests)
    monkeypatch.setattr(
        g_infer_context,
        "radix_cache",
        SimpleNamespace(
            get_big_page_ids_by_node=lambda node: [],
            free_one_small_page_buffer=lambda: None,
            small_page_buffers=coordinator.small_page_buffers,
        ),
    )
    yield mem, requests, coordinator, config
    get_env_start_args.cache_clear()
    get_unique_server_name.cache_clear()


@pytest.mark.parametrize("token_num", [320, 384])
def test_cpu_roundtrip_keeps_kv_continuous_and_restores_only_saved_states(
    monkeypatch, linear_cache, cpu_cache_factory, token_num
):
    from lightllm.server.router.dynamic_prompt.hybrid_att_radix_cache import HybridAttPagedRadixCache
    from lightllm.server.router.model_infer.mode_backend import multi_level_kv_cache as common
    from lightllm.server.router.model_infer.mode_backend.multi_level_kv_cache import MultiLevelKvCacheModule

    mem, requests, coordinator, config = linear_cache
    disable_chunked_mm_input = get_env_start_args().disable_chunked_mm_input
    first_page_has_state = not disable_chunked_mm_input
    big = coordinator.big_page_buffers
    big_states = {}
    for end in range(128, token_num + 1, 128):
        slot = big.alloc_one_state_cache()
        big_states[end] = slot
        for tensor in big.get_state_cache(slot):
            tensor.normal_()
    tail_slot = None
    if token_num % 128:
        tail_slot = coordinator.small_page_buffers.alloc_one_state_cache()
        for tensor in coordinator.small_page_buffers.get_state_cache(tail_slot):
            tensor.normal_()
        final_states = coordinator.small_page_buffers.get_state_cache(tail_slot)
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

    coordinator.offload_gpu_kv_to_cpu_cache(
        indexes, page_ids, torch.zeros_like(page_ids), client, req, g_infer_context.radix_cache
    )
    torch.cuda.synchronize()
    module = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    module.cpu_cache_client = client
    client.update_pages_status_to_ready(pages, deref=False)
    big.free_state_cache(list(big_states.values()))
    if tail_slot is not None:
        coordinator.small_page_buffers.free_state_cache([tail_slot])
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
        model=SimpleNamespace(mem_manager=mem, req_manager=requests, cache_coordinator=coordinator),
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
    torch.testing.assert_close(coordinator.linear_state_manager.conv_state.buffer[:, 0].cpu(), expected_states[0])
    torch.testing.assert_close(coordinator.linear_state_manager.ssm_state.buffer[:, 0].cpu(), expected_states[1])

    # Loaded intermediate pages must remain unusable as radix restore points.
    tree = HybridAttPagedRadixCache(384, 103, 64, 2, mem, coordinator.small_page_buffers, coordinator.big_page_buffers)
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


def test_two_tail_loads_restore_before_reusing_temporary_slot(monkeypatch, linear_cache, cpu_cache_factory):
    from lightllm.server.router.model_infer.mode_backend import multi_level_kv_cache as common
    from lightllm.server.router.model_infer.mode_backend.multi_level_kv_cache import MultiLevelKvCacheModule

    mem, requests, coordinator, config = linear_cache
    client = cpu_cache_factory(
        CpuKVCacheMeta(4, 1, 1, 1, config.get_cpu_cache_big_page_bytes(), torch.uint8, 0, torch.uint8),
    )
    reqs, expected = [], []
    for index in range(2):
        big_slot = coordinator.big_page_buffers.alloc_one_state_cache()
        small_slot = coordinator.small_page_buffers.alloc_one_state_cache()
        for state in coordinator.big_page_buffers.get_state_cache(big_slot):
            state.fill_(index + 10)
        states = coordinator.small_page_buffers.get_state_cache(small_slot)
        for state in states:
            state.fill_(index + 1)
        expected.append([state.clone() for state in states])
        pages, _ = client.allocate_pages([100 + index * 2, 101 + index * 2], disk_offload_enable=False)
        req = SimpleNamespace(
            req_idx=index,
            cur_kv_len=192,
            hold_kv_len=192,
            hybrid_cache_len=192,
            hybrid_len_to_big_page_id={128: big_slot},
            tail_small_page_buffer_id=small_slot,
            shared_kv_node=None,
            mm_input_spans=[],
            sampling_param=SimpleNamespace(shm_param=SimpleNamespace(prompt_logprobs=-1)),
            shm_req=SimpleNamespace(
                input_len=256,
                disk_prompt_cache_len=0,
                cpu_cache_match_page_indexes=SimpleNamespace(get_all=lambda pages=pages: pages),
                token_hash_page_len_list=SimpleNamespace(get_all=lambda: [128, 192]),
            ),
        )
        indexes = torch.arange(index * 192, (index + 1) * 192, dtype=torch.int32, device="cuda")
        mem.kv_buffer[:, index * 192 : (index + 1) * 192].fill_(index + 1)
        page_ids = torch.tensor(pages, dtype=torch.int32, device="cuda")
        coordinator.offload_gpu_kv_to_cpu_cache(
            indexes, page_ids, torch.zeros_like(page_ids), client, req, g_infer_context.radix_cache
        )
        torch.cuda.synchronize()
        client.update_pages_status_to_ready(pages, deref=False)
        coordinator.big_page_buffers.free_state_cache([big_slot])
        coordinator.small_page_buffers.free_state_cache([small_slot])
        req.hybrid_len_to_big_page_id = {}
        req.tail_small_page_buffer_id = None
        req.cur_kv_len = req.hold_kv_len = 0
        reqs.append(req)

    mem.kv_buffer.zero_()

    def allocate(req, count):
        allocated = mem.alloc(count)
        requests.req_to_token_indexs[req.req_idx, :count] = allocated.cuda()
        req.hold_kv_len = count
        return allocated

    module = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    module.cpu_cache_client = client
    module.args = get_env_start_args()
    module.backend = SimpleNamespace(
        is_master_in_dp=True,
        _alloc_req_kv_mem=allocate,
        model=SimpleNamespace(mem_manager=mem, req_manager=requests, cache_coordinator=coordinator),
    )
    module.init_sync_group = None
    module.need_sync_compute_stream = lambda: False
    monkeypatch.setattr(common.g_infer_context, "get_can_alloc_token_num", lambda: 384)
    monkeypatch.setattr(common.dist, "barrier", lambda **kwargs: None)
    events = []
    load = coordinator.load_cpu_cache_to_gpu
    restore = coordinator.restore_state

    def track_load(**kwargs):
        events.append(("copy", kwargs["req"].req_idx))
        assert load(**kwargs) is None

    def track_restore(req_idx, pool, slot, restored_len=None):
        assert slot == coordinator.CPU_CACHE_BIG_PAGE_LOAD_TEMP_BUFFER_ID
        events.append(("restore", req_idx))
        restore(req_idx, pool, slot, restored_len)

    monkeypatch.setattr(coordinator, "load_cpu_cache_to_gpu", track_load)
    monkeypatch.setattr(coordinator, "restore_state", track_restore)
    module.load_cpu_cache_to_reqs(reqs)
    assert events == [("copy", 0), ("restore", 0), ("copy", 1), ("restore", 1)]
    for index, req in enumerate(reqs):
        state = coordinator.linear_state_manager
        torch.testing.assert_close(state.conv_state.buffer[:, req.req_idx].cpu(), expected[index][0])
        torch.testing.assert_close(state.ssm_state.buffer[:, req.req_idx].cpu(), expected[index][1])
        assert req.cur_kv_len == req.hold_kv_len == 192
        assert (mem.kv_buffer[:, index * 192 : (index + 1) * 192] == index + 1).all()


def test_ipc_export_keeps_checkpoints_pinned_and_state_separate(linear_cache):
    from multiprocessing.reduction import ForkingPickler
    from lightllm.common.kv_transfer_resources import KVTransferResources, KVTransferView

    mem, requests, coordinator, _ = linear_cache
    pools = (coordinator.big_page_buffers, coordinator.small_page_buffers)
    tensors = [tensor for pool in pools for tensor in (pool.conv_state_cache.buffer, pool.ssm_state_cache.buffer)]
    pointers = [tensor.data_ptr() for tensor in tensors]
    assert all(tensor.is_pinned() for tensor in tensors)
    mem.unrelated_resource = lambda: None
    resources = KVTransferResources(
        KVTransferView.from_mem_manager(mem), requests.req_to_token_indexs, coordinator.create_state_transfer()
    )
    assert len(ForkingPickler.dumps(resources)) > 0
    assert all(tensor.is_pinned() for tensor in tensors)
    assert [tensor.data_ptr() for tensor in tensors] == pointers
    assert not hasattr(resources.kv, "big_page_buffers")
    assert not hasattr(resources.kv, "allocator")
    assert not hasattr(resources.kv, "operator")
    assert not hasattr(resources.kv, "req_to_token_indexs")
    assert not hasattr(resources.kv, "req_to_conv_state")
    assert resources.req_to_token_indexes.data_ptr() == requests.req_to_token_indexs.data_ptr()
    assert (
        resources.state_transfer.req_to_ssm_state.buffer.data_ptr()
        == coordinator.linear_state_manager.ssm_state.buffer.data_ptr()
    )
