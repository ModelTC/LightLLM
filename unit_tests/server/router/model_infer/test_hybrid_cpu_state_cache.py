from unit_tests.common.att_resources.model_helpers import build_hybrid_resources
import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.att_resources.allocators import FullAttKVAllocator

from lightllm.common.att_resources.full_att import (
    GQAFullAttentionState,
    FullAttentionConfig,
)
from lightllm.common.req_manager import ReqManager
from lightllm.server.router.model_infer.mode_backend.multi_level_kv_cache import MultiLevelKvCacheModule
from lightllm.server.multi_level_kv_cache import CacheTier
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.models.qwen3next.config import Qwen3NextResourceConfig
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.envs_utils import get_env_start_args, get_unique_server_name
from lightllm.utils.kv_cache_utils import CpuKVCacheMeta


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.usefixtures("kv_allocator_environment"),
]


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
    config = Qwen3NextResourceConfig(
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
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.float32,
        layer_types=("linear_attention", "full_attention"),
    )
    full = GQAFullAttentionState(
        FullAttentionConfig(tuple((layer,) for layer in config.get_full_att_logical_layer_ids()), 1, 8, torch.bfloat16),
         page_size=args.page_size,
    )
    full.initialize_storage(384)
    requests = ReqManager(2, 512)
    resources = build_hybrid_resources(
        config=config, request_capacity=3, mtp_step=0,
        full_state=full,
        full_allocator=FullAttKVAllocator(full.capacity, full.page_size),
        big_page_capacity=5,
        small_page_capacity=2,
    )
    monkeypatch.setattr(g_infer_context, "att_resource_manager", resources)
    monkeypatch.setattr(g_infer_context, "is_hybrid_att_model", True)
    monkeypatch.setattr(g_infer_context, "req_manager", requests)
    monkeypatch.setattr(
        g_infer_context,
        "radix_cache",
        SimpleNamespace(
            get_big_page_ids_by_node=lambda node: [],
            free_one_small_page_buffer=lambda: None,
        ),
    )
    yield full, resources, requests, config
    get_env_start_args.cache_clear()
    get_unique_server_name.cache_clear()


@pytest.mark.parametrize("token_num", [320, 384])
def test_cpu_roundtrip_requires_all_linear_pages_and_restores_final_state(
    monkeypatch, linear_cache, cpu_cache_factory, token_num
):
    from lightllm.server.router.dynamic_prompt.hybrid_att_radix_cache import HybridAttPagedRadixCache
    from lightllm.server.router.model_infer.mode_backend import multi_level_kv_cache as common
    from lightllm.server.router.model_infer.mode_backend.multi_level_kv_cache import MultiLevelKvCacheModule

    full, resources, requests, config = linear_cache
    disable_chunked_mm_input = get_env_start_args().disable_chunked_mm_input
    first_page_has_state = not disable_chunked_mm_input
    linear = resources.main_gdn_conv_bf16_ssm_fp32_state
    big = linear.big_pages_pool
    small = linear.small_pages_pool
    big_states = {}
    for end in range(128, token_num + 1, 128):
        slot = resources.bounded_att_big_page_allocator.alloc(1)[0]
        big_states[end] = slot
        for tensor in (big.conv_state[slot], big.ssm_state[slot]):
            tensor.normal_()
    tail_slot = None
    if token_num % 128:
        tail_slot = resources.bounded_att_small_page_allocator.alloc(1)[0]
        for tensor in (small.conv_state[tail_slot], small.ssm_state[tail_slot]):
            tensor.normal_()
        final_states = (small.conv_state[tail_slot], small.ssm_state[tail_slot])
    else:
        final_states = (big.conv_state[big_states[token_num]], big.ssm_state[big_states[token_num]])
    expected_states = [tensor.clone() for tensor in final_states]
    expected_big_states = {
        end: [t.clone() for t in (big.conv_state[slot], big.ssm_state[slot])] for end, slot in big_states.items()
    }
    full.k_storage.normal_()
    full.v_storage.normal_()
    expected_k = full.k_storage[:, :token_num].clone()
    expected_v = full.v_storage[:, :token_num].clone()
    client = cpu_cache_factory(
        CpuKVCacheMeta(3, resources.get_cpu_page_bytes()),
    )
    indexes = torch.arange(token_num, dtype=torch.int32, device="cuda")
    req = SimpleNamespace(
        req_idx=0,
        cache_tiers=(CacheTier.CPU,),
        cpu_cache_task_status=InferReq._CpuCacheTaskStatus.NOT_STARTED,
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
            token_hash_list=SimpleNamespace(get_all=lambda: [11, 22, 33]),
            token_hash_page_len_list=SimpleNamespace(get_all=lambda: [128, 256, token_num]),
        ),
    )

    # 通过接收 InferReq 的生产入口完成页来源查询、CPU 分配与快照写出。
    requests.req_to_token_indexs[0, :token_num] = indexes
    module = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    module.args = get_env_start_args()
    module.cpu_cache_client = client
    module.gloo_group = None
    module.backend = SimpleNamespace(
        is_master_in_dp=True, is_hybrid_att_model=True, dp_world_size=1,
        model=SimpleNamespace(req_manager=requests, att_resource_manager=resources),
    )
    monkeypatch.setattr(common.dist, "broadcast_object_list", lambda *args, **kwargs: None)
    task = module.write_req_to_cpu(req, torch.cuda.Stream())
    assert task is not None
    task.sync_event.synchronize()
    pages = task.page_indexes.tolist()
    client.update_pages_status_to_ready(pages, deref=False)
    resources.bounded_att_big_page_allocator.free(list(big_states.values()))
    if tail_slot is not None:
        resources.bounded_att_small_page_allocator.free([tail_slot])
    req.hybrid_len_to_big_page_id = {}
    req.tail_small_page_buffer_id = None
    req.cur_kv_len = req.hold_kv_len = 0
    full.k_storage.zero_()
    full.v_storage.zero_()
    big.conv_state.zero_()
    big.ssm_state.zero_()

    def allocate(req, count):
        allocated = resources.full_att_allocator.alloc(count)
        requests.req_to_token_indexs[req.req_idx, :count] = allocated.cuda()
        req.hold_kv_len = count
        return allocated

    module.args = get_env_start_args()
    module.backend = SimpleNamespace(
        is_master_in_dp=True,
        _alloc_req_kv_mem=allocate,
        model=SimpleNamespace(req_manager=requests, att_resource_manager=resources),
        is_hybrid_att_model=True,
    )
    module.init_sync_group = None
    module.need_sync_compute_stream = lambda: False
    monkeypatch.setattr(common.g_infer_context, "get_can_alloc_token_num", lambda: 384)
    monkeypatch.setattr(common.dist, "barrier", lambda **kwargs: None)
    module.load_cpu_cache_to_reqs([req])

    assert req.cur_kv_len == req.hold_kv_len == token_num
    restored_indices = requests.req_to_token_indexs[req.req_idx, :token_num]
    assert torch.equal(full.k_storage[:, restored_indices], expected_k)
    assert torch.equal(full.v_storage[:, restored_indices], expected_v)
    assert sorted(req.hybrid_len_to_big_page_id) == list(range(128, token_num + 1, 128))
    # Every published CPU page has a complete, length-checked linear state.
    for end, slot in req.hybrid_len_to_big_page_id.items():
        for actual, expected in zip((big.conv_state[slot], big.ssm_state[slot]), expected_big_states[end]):
            torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(linear.conv_state[:, 0].cpu(), expected_states[0])
    torch.testing.assert_close(linear.ssm_state[:, 0].cpu(), expected_states[1])

    # Model data validity and multimodal resume boundaries are separate checks.
    tree = HybridAttPagedRadixCache(384, 103, 64, 2, resources)
    cache_len = max(req.hybrid_len_to_big_page_id)
    tree.insert(
        torch.arange(cache_len),
        value=restored_indices[:cache_len].cpu(),
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


@pytest.mark.parametrize("shared_prefix", [False, True])
@pytest.mark.parametrize("allocated_pages", [0, 1, 2])
def test_offload_resolves_only_allocated_pages_and_requires_big_slots(
    monkeypatch, linear_cache, shared_prefix, allocated_pages,
):
    from lightllm.server.router.model_infer.mode_backend import multi_level_kv_cache as common

    full, resources, requests, config = linear_cache
    slots = resources.bounded_att_big_page_allocator.alloc(allocated_pages) if allocated_pages else []
    shared_slots = [slots[0]] if shared_prefix and slots else []
    monkeypatch.setattr(g_infer_context, "radix_cache", SimpleNamespace(get_big_page_ids_by_node=lambda node: shared_slots))
    big_slots = {128 * (i + 1): slot for i, slot in enumerate(slots) if not shared_prefix or i > 0}
    req = SimpleNamespace(
        req_idx=0, shared_kv_node=None, hybrid_len_to_big_page_id=big_slots,
        tail_small_page_buffer_id=None, cur_kv_len=256, cache_tiers=(CacheTier.CPU,),
        shm_req=SimpleNamespace(
            token_hash_list=SimpleNamespace(get_all=lambda: [11, 22]),
            token_hash_page_len_list=SimpleNamespace(get_all=lambda: [128, 256]),
        ),
    )
    requested_hashes = []
    writes = []

    def allocate_pages(hashes, *, disk_offload_enable):
        requested_hashes.append(list(hashes))
        return list(range(allocated_pages)) + [-1] * (len(hashes) - allocated_pages), [False] * len(hashes)

    def write_to_cpu(token_indices, cpu_page_buffers, cpu_page_indices, **kwargs):
        writes.append((token_indices.numel(), cpu_page_indices, kwargs["big_page_slot_ids"], kwargs["small_page_slot_ids"]))

    module = MultiLevelKvCacheModule.__new__(MultiLevelKvCacheModule)
    module.args = get_env_start_args()
    module.gloo_group = None
    module.backend = SimpleNamespace(
        is_master_in_dp=True, is_hybrid_att_model=True, dp_world_size=1,
        model=SimpleNamespace(att_resource_manager=resources, req_manager=requests),
    )
    module.cpu_cache_client = SimpleNamespace(
        lock=SimpleNamespace(acquire_sleep1ms=lambda: None, release=lambda: None),
        allocate_pages=allocate_pages, cpu_kv_cache_tensor=torch.empty((2, 1), dtype=torch.uint8),
    )
    monkeypatch.setattr(resources, "write_req_to_cpu", write_to_cpu)
    broadcasts = []
    monkeypatch.setattr(common.dist, "broadcast_object_list", lambda data, **kwargs: broadcasts.append(data))
    task = module.write_req_to_cpu(req, torch.cuda.Stream())
    if allocated_pages == 0:
        assert task is None
        assert req.cpu_cache_task_status == InferReq._CpuCacheTaskStatus.FINISHED
        assert requested_hashes == [[11, 22]]
        assert broadcasts == [[0]]
        assert writes == []
        return
    task.sync_event.synchronize()
    assert broadcasts == [[{
        "item_size": allocated_pages,
        "page_list": list(range(allocated_pages)),
        "ready_list": [False] * allocated_pages,
    }]]
    # 即使 CPU 只分配一页，也申请完整可缓存范围；未选中的 256 位置不要求提供槽位。
    assert requested_hashes == [[11, 22]]
    assert writes == [(allocated_pages * 128, list(range(allocated_pages)), slots, [])]

    # 实际选中页缺失时直接暴露错误，不缩短写出范围，也不提交部分搬运。
    missing_length = allocated_pages * 128
    if shared_prefix and allocated_pages == 1:
        shared_slots.clear()
    else:
        del req.hybrid_len_to_big_page_id[missing_length]
    with pytest.raises(KeyError) as error:
        module.write_req_to_cpu(req, torch.cuda.Stream())
    assert error.value.args == (missing_length,)
    assert requested_hashes == [[11, 22], [11, 22]]
    assert len(writes) == 1
