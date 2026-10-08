from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from sortedcontainers import SortedSet

from lightllm.models.deepseek_v4.triton_kernel import dp_cache_io
from lightllm.server.core.objs import StartArgs
from lightllm.server.router.dynamic_prompt.hybrid_att_radix_cache import (
    HybridAttPagedRadixCache,
    HybridAttPagedTreeNode,
)
from lightllm.server.router.model_infer.infer_batch import InferenceContext
from lightllm.server.router.model_infer.mode_backend.dp_backend import dp_shared_kv_trans as transfer, impl
from lightllm.utils import config_utils


@pytest.mark.parametrize("dsv4", [False, True])
def test_only_generic_path_allocates_shared_request_table(monkeypatch, dsv4):
    shared = Mock()
    monkeypatch.setattr(transfer, "ShmArray", shared)
    monkeypatch.setattr(transfer, "get_env_start_args", lambda: NS(diverse_mode=False))
    monkeypatch.setattr(transfer, "get_dp_rank_in_node", lambda: 0)
    monkeypatch.setattr("lightllm.utils.device_utils.kv_trans_use_p2p", lambda: True)
    module = transfer.DPKVSharedMoudle(9, 4, NS(is_deepseek_v4=dsv4))
    if dsv4:
        shared.assert_not_called()
        assert not hasattr(module, "shared_req_infos")
    else:
        shared.assert_called_once_with(name="dp_shared_req_infos", shape=(9, 4, 2), dtype=np.int64)
        shared.return_value.create_shm.assert_called_once()


def test_checkpoint_pins_survive_eviction_and_misses_balance_refs():
    cache = object.__new__(HybridAttPagedRadixCache)
    cache.hash_page_size, cache.big_page_num, cache.big_page_tokens = 256, 8, 2048
    cache._evict_tree_set = SortedSet(key=lambda node: node.get_compare_key())
    cache._evict_tree_set_for_state_cache = SortedSet(key=lambda node: node.get_compare_key_for_buffer_idx())
    cache.refed_tokens_num = NS(arr=np.zeros(1, dtype=np.int64))
    freed = []
    cache.small_page_buffers = NS(
        get_free_cache_num=lambda: 0, free_state_cache=lambda free_indexes: freed.extend(free_indexes)
    )
    root = cache.root_node = HybridAttPagedTreeNode(256, 8)
    root.page_num, root.ref_counter = 8, 1
    root.token_mem_index_value = torch.empty(0, dtype=torch.int32)
    key = torch.arange(256)
    child = root.add_and_return_new_child(key, key.int(), 17, 0)
    cache._add_node(child)

    for _ in range(2):
        node, length, value = cache.match_prefix(key, [17], True, False, True)
        assert node is child and length == 256 and value is None
    cache.free_one_small_page_buffer()
    assert freed == [] and child.checkpoint_transfer_refs == 2
    cache.release_checkpoint_pin(child)
    cache.dec_node_ref_counter(child)
    cache.free_one_small_page_buffer()
    assert freed == []
    cache.release_checkpoint_pin(child)
    cache.dec_node_ref_counter(child)
    cache.free_one_small_page_buffer()
    assert freed == [0]
    for hashes in ([17], [99]):
        assert cache.match_prefix(key, hashes, True, False, True) == (None, 0, None)
    assert root.ref_counter == 1 and child.ref_counter == 0


@pytest.mark.parametrize("image_span,expected", [((2700, 2900), 2560), ((2816, 2900), 2816), ((2600, 2816), 2816)])
def test_vision_matching_releases_rejected_checkpoint(image_span, expected):
    first, second = NS(node_prefix_total_len=2816), NS(node_prefix_total_len=2560)
    cache = NS(
        match_prefix=Mock(side_effect=[(first, 2816, None), (second, 2560, None)]),
        release_checkpoint_pin=Mock(),
        dec_node_ref_counter=Mock(),
    )
    context = object.__new__(InferenceContext)
    context.args, context.radix_cache = NS(linear_att_hash_page_size=256), cache
    req = NS(
        sample_params=NS(disable_prompt_cache=False),
        hybrid_token_hash_list=NS(get_all=lambda: list(range(11))),
        shm_prompt_ids=NS(arr=np.arange(3000)),
    )
    node = context.retain_hybrid_prefix(req, 3000, [image_span], pin_checkpoint=True)
    assert node.node_prefix_total_len == expected
    if expected == 2560:
        cache.release_checkpoint_pin.assert_called_once_with(first)
        cache.dec_node_ref_counter.assert_called_once_with(first)
        assert len(cache.match_prefix.call_args.args[0]) == 2560
    else:
        cache.release_checkpoint_pin.assert_not_called()


def test_radix_extracts_suffix_across_big_page():
    cache = object.__new__(HybridAttPagedRadixCache)
    root = HybridAttPagedTreeNode(256, 8)
    root.token_mem_index_value = torch.empty(0, dtype=torch.int32)
    big = root.add_and_return_new_big_page_child(torch.arange(2048), torch.arange(2048).int(), 17, 0)
    tail = big.add_and_return_new_child(torch.arange(2048, 2304), torch.arange(2048, 2304).int(), 19, 1)
    assert torch.equal(cache.get_mem_index_value_by_node(tail, 1792, 2304), torch.arange(1792, 2304))


def test_probe_uses_shm_and_radix_only_and_releases_on_failure(monkeypatch):
    cache = NS(release_checkpoint_pin=Mock(), dec_node_ref_counter=Mock())
    shm = NS(input_len=3000, link_prompt_ids_shm_array=Mock())
    context = NS(
        shm_req_manager=NS(get_req_obj_by_index=Mock(return_value=shm), put_back_req_obj=Mock()),
        retain_hybrid_prefix=Mock(side_effect=[NS(node_prefix_total_len=2816), RuntimeError("probe failed")]),
    )
    monkeypatch.setattr(transfer, "g_infer_context", context)
    module = object.__new__(transfer.DPKVSharedMoudle)
    module.backend = NS(radix_cache=cache)
    params = NS(to_dict=lambda: {"images": []})
    with pytest.raises(RuntimeError, match="probe failed"):
        module.probe_dsv4_matches([(7, 0, params, 0), (8, 1, params, 0)])
    assert context.shm_req_manager.put_back_req_obj.call_count == 2
    cache.release_checkpoint_pin.assert_called_once()
    cache.dec_node_ref_counter.assert_called_once()


def test_only_owner_requests_are_created_and_foreign_pins_are_released(monkeypatch):
    events = []
    foreign = (8, 1, None, 1)
    module = NS(
        probe_dsv4_matches=Mock(return_value={8: NS(release=lambda: events.append("release"))}),
        gather_dsv4_match_lens=Mock(),
        build_dsv4_trans_tasks=Mock(return_value=([], [])),
        kv_trans_dsv4=Mock(side_effect=RuntimeError("transfer failed")),
    )
    context = NS(add_reqs=Mock(return_value=[]))
    monkeypatch.setattr(impl, "g_infer_context", context)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: NS(synchronize=lambda: events.append("sync")))
    backend = object.__new__(impl.DPChunkedPrefillBackend)
    backend.args, backend.dp_rank_in_node = NS(enable_dp_prompt_cache_fetch=True), 0
    backend.is_deepseek_v4, backend.dp_kv_shared_module = True, module
    with pytest.raises(RuntimeError, match="transfer failed"):
        backend._init_reqs([(7, 0, None, 0), foreign])
    module.probe_dsv4_matches.assert_called_once_with([foreign])
    context.add_reqs.assert_called_once_with([(7, 0, None, 0)], init_prefix_cache=True)
    assert events == ["sync", "release"]


def test_source_advertisement_needs_no_runtime_capacity_and_rejects_tp_disagreement(monkeypatch):
    module = object.__new__(transfer.DPKVSharedMoudle)
    module.dp_rank_in_node = 0
    module.backend = NS(node_gloo_group=object(), dp_size_in_node=2, dp_world_size=2)
    monkeypatch.setattr(transfer, "g_infer_context", object())
    monkeypatch.setattr(transfer.dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(
        transfer.dist,
        "all_gather_object",
        lambda output, local, group: output.__setitem__(slice(None), [local, [0, 512], [0, 0], [0, 0]]),
    )
    matches = {8: NS(matched_len=2816)}
    lengths = module.gather_dsv4_match_lens([(7, 0, None, 0), (8, 1, None, 1)], matches)
    assert lengths.tolist() == [[0, 0], [0, 0]]
    monkeypatch.setattr(
        transfer.dist,
        "all_gather_object",
        lambda output, local, group: output.__setitem__(
            slice(None), [local, local, [0] * len(local), [0] * len(local)]
        ),
    )
    assert module.gather_dsv4_match_lens([(8, 1, None, 1)], matches)[0, 0] == 2816


@pytest.fixture
def destination(monkeypatch):
    req = NS(
        req_id=7,
        cur_kv_len=0,
        hold_kv_len=0,
        tail_small_page_buffer_id=None,
        hybrid_cache_len=2816,
        hybrid_len_to_big_page_id={},
        get_cur_total_len=lambda: 3000,
        _kv_cache_alloc_need=lambda end: end,
    )

    def allocate(local_req, count):
        local_req.hold_kv_len += count
        return torch.arange(count, dtype=torch.int32)

    module = object.__new__(transfer.DPKVSharedMoudle)
    module.dp_rank_in_node = 0
    module.backend = NS(
        node_gloo_group=object(),
        dp_size_in_node=2,
        dp_world_size=1,
        rank_in_dp=0,
        args=NS(linear_att_hash_page_size=256, linear_att_page_block_num=8, max_req_total_len=4096),
        model=NS(
            req_manager=NS(get_prompt_cache_page_size=lambda: 256),
            mem_manager=NS(
                big_page_buffers=NS(get_free_cache_num=lambda: 1, alloc_one_state_cache=Mock(return_value=4)),
                swa_pool=NS(page_size=128),
            ),
        ),
        small_page_buffers=NS(alloc_one_state_cache=Mock(return_value=3), free_state_cache=Mock()),
        radix_cache=NS(get_available_small_page_buffer_num=lambda: 1, free_one_small_page_buffer=Mock()),
        _alloc_req_kv_mem=Mock(side_effect=allocate),
    )
    context = NS(get_can_alloc_dsv4_swa_page_num=lambda: 2, get_can_alloc_token_num=lambda: 4096)
    monkeypatch.setattr(transfer, "g_infer_context", context)
    monkeypatch.setattr(transfer.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        transfer.dist, "all_gather_object", lambda output, local, group: output.__setitem__(slice(None), [local, {}])
    )
    return module, req, context


@pytest.mark.parametrize("resource", ["swa", "tokens", "big", "small", "tp"])
def test_destination_admission_preserves_capacity_and_tp_agreement(destination, monkeypatch, resource):
    module, req, context = destination
    if resource in ("swa", "tokens"):
        setattr(
            context, "get_can_alloc_dsv4_swa_page_num" if resource == "swa" else "get_can_alloc_token_num", lambda: 0
        )
    elif resource == "big":
        module.backend.model.mem_manager.big_page_buffers.get_free_cache_num = lambda: 0
    elif resource == "small":
        module.backend.radix_cache.get_available_small_page_buffer_num = lambda: 0
    else:
        module.backend.dp_world_size = 2
        monkeypatch.setattr(transfer.dist, "get_world_size", lambda group: 4)
        monkeypatch.setattr(
            transfer.dist,
            "all_gather_object",
            lambda output, local, group: output.__setitem__(slice(None), [local, {}, {}, {}]),
        )
    tasks, plan = module.build_dsv4_trans_tasks([(7, 0, None, 0)], [req], np.array([[0, 2816]]))
    assert tasks == [] and all(not tasks for tasks in plan)
    module.backend._alloc_req_kv_mem.assert_not_called()
    module.backend.small_page_buffers.alloc_one_state_cache.assert_not_called()
    assert req.hybrid_len_to_big_page_id == {}


def test_received_checkpoints_remain_owned_for_later_radix_insertion(destination):
    module, req, _ = destination
    tasks, plan = module.build_dsv4_trans_tasks([(7, 0, None, 0)], [req], np.array([[0, 2816]]))
    assert plan == [[(1, 7, 0, 2816)], []]
    assert req.hybrid_len_to_big_page_id == {2048: 4}
    assert req.tail_small_page_buffer_id == 3 and tasks[0].terminal_small_page_buffer_id == 3
    assert len(tasks[0].mem_indexes) == req.hold_kv_len == 2816


def test_failed_allocation_releases_temporary_checkpoint(destination):
    module, req, _ = destination
    module.backend._alloc_req_kv_mem.side_effect = RuntimeError("allocation failed")
    with pytest.raises(RuntimeError, match="allocation failed"):
        module.build_dsv4_trans_tasks([(7, 0, None, 0)], [req], np.array([[0, 2560]]))
    module.backend.small_page_buffers.free_state_cache.assert_called_once_with([3])
    assert req.tail_small_page_buffer_id is None


@pytest.mark.parametrize("start,end", [(0, 2816), (0, 2048), (2048, 2560)])
def test_source_sends_history_indexes_and_required_checkpoints_without_inferreq(monkeypatch, start, end):
    big = torch.arange(16, dtype=torch.uint8).view(4, 4)
    small = torch.full((1, 4), 9, dtype=torch.uint8)
    node = NS(node_prefix_total_len=end, small_page_buffer_idx=0)
    cache = NS(
        get_mem_index_value_by_node=Mock(side_effect=lambda node, first, last: torch.arange(first, last).int()),
        get_big_page_ids_by_node=lambda node: [0, 1, 2, 3],
    )
    module = object.__new__(transfer.DPKVSharedMoudle)
    module.backend = NS(
        args=NS(linear_att_hash_page_size=256, linear_att_page_block_num=8, max_req_total_len=4096),
        node_gloo_group=object(),
        radix_cache=cache,
        model=NS(mem_manager=NS(big_page_buffers=NS(buffer=big))),
        small_page_buffers=NS(buffer=small),
    )
    sent = []
    monkeypatch.setattr(transfer, "g_infer_context", object())
    monkeypatch.setattr(transfer.dist, "get_rank", lambda group: 1)
    monkeypatch.setattr(transfer.dist, "get_global_rank", lambda group, rank: rank)
    monkeypatch.setattr(transfer.dist, "send", lambda tensor, dst, group: sent.append(tensor.clone()))
    module._transfer_dsv4_source_data([], {7: transfer.PrefixCacheMatch(node, cache)}, [[(1, 7, start, end)], []])
    assert torch.equal(sent[0], torch.arange(start, end).int()) and sent[0].dtype == torch.int32
    expected = [big[length // 2048 - 1] for length in range((start // 2048 + 1) * 2048, end + 1, 2048)]
    if end % 2048:
        expected.append(small[0])
    assert len(sent) == len(expected) + 1
    assert all(torch.equal(actual, expected) for actual, expected in zip(sent[1:], expected))


@pytest.mark.parametrize("receives", [False, True])
@pytest.mark.parametrize("cpu_cache", [False, True])
def test_history_only_restore_finishes_before_all_rank_fence(monkeypatch, receives, cpu_cache):
    events = []
    table = torch.arange(512, dtype=torch.int32).view(2, 256)
    req = NS(req_id=7, req_idx=1, cur_kv_len=0, hold_kv_len=256, shm_req=NS())
    task = transfer.TransTask(req, table[1], 1, 1, table[0], terminal_small_page_buffer_id=3)
    req_manager = NS(
        req_to_token_indexs=table,
        get_prompt_cache_page_size=lambda: 256,
        restore_state=Mock(side_effect=lambda *args, **kwargs: events.append("restore")),
    )
    module = object.__new__(transfer.DPKVSharedMoudle)
    module.dp_rank_in_node = 0
    module.backend = NS(
        args=NS(linear_att_hash_page_size=256, linear_att_page_block_num=8, enable_cpu_cache=cpu_cache),
        model=NS(mem_manager=object()),
        small_page_buffers=object(),
        node_nccl_group=object(),
        is_master_in_dp=True,
        logger=NS(info=lambda message: None),
    )
    module.dsv4_source_pool_ptrs = torch.zeros(7, dtype=torch.uint64)
    module._transfer_dsv4_source_data = lambda *args: events.append("source_data")
    monkeypatch.setattr(transfer, "g_infer_context", NS(req_manager=req_manager))
    monkeypatch.setattr(
        transfer, "g_pin_mem_manager", NS(gen_from_list=lambda key, data, dtype: torch.tensor(data, dtype=dtype))
    )
    copies = Mock(side_effect=lambda **kwargs: events.append("history"))
    monkeypatch.setattr(transfer, "copy_dsv4_dp_caches", copies)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: NS(synchronize=lambda: events.append("sync")))
    monkeypatch.setattr(transfer.dist, "barrier", lambda group: events.append("barrier"))
    module.kv_trans_dsv4([task] if receives else [], {}, [])
    assert events[-2:] == ["sync", "barrier"]
    if receives:
        assert events == ["source_data", "history", "restore", "sync", "barrier"]
        assert copies.call_args.kwargs["copy_runtime"] is False
        req_manager.restore_state.assert_called_once_with(req, module.backend.small_page_buffers, 3, checkpoint_len=256)
        assert req.cur_kv_len == req.shm_req.shm_cur_kv_len == req.shm_req.prompt_cache_len == 256
    else:
        copies.assert_not_called()


def test_history_only_kernel_has_no_runtime_programs(monkeypatch):
    grids = []

    class Kernel:
        def __getitem__(self, grid):
            grids.append(grid)
            return lambda *args, **kwargs: None

    pool = NS(buffer=torch.empty((1, 1, 1), dtype=torch.uint8), page_size=1, bytes_per_page=1)
    manager = NS(
        n_c4=1,
        n_c128=0,
        layer_num=1,
        c4_pool=pool,
        c4_indexer_pool=pool,
        c128_pool=None,
        swa_pool=pool,
        c4_state_buffer=torch.empty((1, 1, 4)),
        c4_indexer_state_buffer=torch.empty((1, 1, 4)),
        req_to_swa_pages=torch.empty((1, 2), dtype=torch.int32),
        c4_state_ring=4,
    )
    monkeypatch.setattr(dp_cache_io, "_copy_dsv4_dp_caches_kernel", Kernel())
    dp_cache_io.copy_dsv4_dp_caches(
        torch.zeros(7, dtype=torch.uint64),
        manager,
        torch.zeros(6, dtype=torch.uint64),
        torch.zeros(2, dtype=torch.uint64),
        copy_runtime=False,
    )
    assert grids == [(1,)]


@pytest.mark.parametrize(
    "model,fetch,diverse,expected",
    [
        ("deepseek_v4", True, False, 3),
        ("deepseek_v4", False, False, 3),
        ("deepseek_v4", True, True, 9),
        ("glm5_next", True, False, 9),
        ("llama", True, False, 9),
    ],
)
def test_request_capacity_keeps_generic_and_beam_contracts(monkeypatch, model, fetch, diverse, expected):
    monkeypatch.setattr(config_utils, "get_model_type", lambda path: model)
    monkeypatch.setattr(config_utils, "is_hybrid_att_model", lambda path: model != "llama")
    args = StartArgs(running_max_req_size=9, dp=8, nnodes=2, enable_dp_prompt_cache_fetch=fetch, diverse_mode=diverse)
    assert config_utils.get_running_max_req_size_per_dp(args) == expected
