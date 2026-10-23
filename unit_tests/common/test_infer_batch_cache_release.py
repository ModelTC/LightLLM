"""Exercise request release with real radix trees, KV allocation and checkpoint slot pools."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from sortedcontainers import SortedDict

from lightllm.server.router.model_infer.mode_backend import generic_pre_process  # noqa: F401
from lightllm.common.kv_cache_mem_manager import allocator as allocator_impl
from lightllm.common.state_cache_manager.base import StateCacheManager
from lightllm.server.multi_level_kv_cache import CacheTier
from lightllm.server.router.dynamic_prompt import hybrid_att_radix_cache, radix_cache
from lightllm.server.router.model_infer import infer_batch
from lightllm.server.router.model_infer.infer_req_ext1 import PinMemPageCacheReqTokenIndex, create_req_token_index


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires pinned memory and CUDA index tables")


class _LocalSharedArray:
    def __init__(self, name, shape, dtype):
        self.arr = np.zeros(shape, dtype=dtype)


class _StateSlots(StateCacheManager):
    def __init__(self):
        super().__init__(size=16)
        self.states = torch.empty((self.size, 1))

    def get_state_cache(self, buffer_idx):
        return self.states[buffer_idx]


@pytest.fixture
def make_release_case(monkeypatch):
    # Only replace interprocess counters; allocation, tree operations and slot accounting are production code.
    monkeypatch.setattr(radix_cache, "SharedArray", _LocalSharedArray)
    monkeypatch.setattr(hybrid_att_radix_cache, "SharedArray", _LocalSharedArray)
    monkeypatch.setattr(allocator_impl, "SharedInt", lambda name: SimpleNamespace(set_value=lambda value: None))
    monkeypatch.setattr(allocator_impl, "get_current_rank_in_node", lambda: 0)
    monkeypatch.setattr(PinMemPageCacheReqTokenIndex, "_page_offsets", None)

    def make_case(attention, mode, page_size, cache_case):
        hybrid = attention != "full"
        partial_prefill = attention == "hybrid_partial"
        allocator = allocator_impl.KvCacheAllocator(64, page_size)
        small_slots, big_slots = _StateSlots(), _StateSlots()
        memory = SimpleNamespace(
            page_size=page_size,
            allocator=allocator,
            alloc=allocator.alloc,
            free=allocator.free,
            big_page_buffers=big_slots,
        )
        if hybrid:
            cache = hybrid_att_radix_cache.HybridAttPagedRadixCache(
                64,
                0,
                hash_page_size=4,
                big_page_num=2,
                kv_cache_mem_manager=memory,
                small_page_buffers=small_slots,
            )
        else:
            cache = radix_cache.RadixCache(64, 0, mem_manager=memory, page_size=page_size)
        key = torch.arange(32, dtype=torch.int64)
        seeded_indexes = None
        if cache_case != "uncached":
            seeded_indexes = allocator.alloc(8).clone()
            if hybrid:
                cache.insert(
                    key[:8],
                    seeded_indexes,
                    block_hashs=[1, 2],
                    block_state_idxs=[None, None],
                    len_to_big_page_id=SortedDict({8: big_slots.alloc_one_state_cache()}),
                )
            else:
                cache.insert(key[:8], seeded_indexes)

        shared_node, shared_indexes = None, None
        if cache_case == "shared":
            if hybrid:
                shared_node, shared_len, shared_indexes = cache.match_prefix(
                    key[:8], block_hashs=[1, 2], update_refs=True
                )
            else:
                shared_node, shared_len, shared_indexes = cache.match_prefix(key[:8], update_refs=True)
            assert shared_len == cache.get_refed_tokens_num() == 8
        shared_len = 0 if shared_node is None else shared_node.node_prefix_total_len
        context = infer_batch.InferenceContext()
        context.args = SimpleNamespace(page_size=page_size, linear_att_hash_page_size=4, linear_att_page_block_num=2)
        context.is_hybrid_att_model = hybrid
        context.radix_cache = cache
        context.req_manager = SimpleNamespace(
            page_size=page_size, req_to_token_indexs=torch.full((1, 32), -1, dtype=torch.int32, device="cuda")
        )
        req = infer_batch.InferReq.__new__(infer_batch.InferReq)
        req.__dict__.update(
            req_idx=0,
            req_manager=context.req_manager,
            args=SimpleNamespace(req_token_index_mode=mode),
            cur_kv_len=12 if partial_prefill else 13,
            hold_kv_len=16,
            cache_tiers=(CacheTier.GPU,),
            shared_kv_node=shared_node,
            hybrid_cache_len=24 if partial_prefill else 12,
            tail_small_page_buffer_id=small_slots.alloc_one_state_cache() if attention == "hybrid_small" else None,
            hybrid_len_to_big_page_id=SortedDict(),
            get_input_token_ids=lambda: key.tolist(),
            shm_req=SimpleNamespace(
                input_len=25 if partial_prefill else 13,
                shm_cur_kv_len=12 if partial_prefill else 13,
                hybrid_token_hash_list=SimpleNamespace(get_all=lambda: [1, 2, 3, 4, 5, 6]),
            ),
        )
        if hybrid and shared_len < 8:
            req.hybrid_len_to_big_page_id[8] = big_slots.alloc_one_state_cache()
        req.token_index = create_req_token_index(req, torch.cuda.current_stream())
        if shared_indexes is not None:
            req.token_index.append_token_indexes(shared_indexes)
        req.token_index.append_token_indexes(allocator.alloc(req.hold_kv_len - shared_len))
        cache_len = (13 // page_size * page_size) if not hybrid else (12 if attention == "hybrid_small" else 8)
        expected_indexes = req.token_index.get_cpu_token_indexes(0, cache_len)
        if seeded_indexes is not None:
            expected_indexes[:8] = seeded_indexes
        monkeypatch.setattr(infer_batch, "g_infer_context", context)
        monkeypatch.setattr(infer_batch, "get_env_start_args", lambda: context.args)
        return context, req, allocator, small_slots, big_slots, expected_indexes

    return make_case


@pytest.mark.parametrize("attention", ["full", "hybrid_small", "hybrid_big", "hybrid_partial"])
@pytest.mark.parametrize("mode", ["pin_mem_cache", "pin_mem_page_cache", "pure_gpu"])
@pytest.mark.parametrize("page_size", [1, 4])
@pytest.mark.parametrize("cache_case", ["uncached", "duplicate", "shared"])
def test_release_and_eviction_conserve_kv_pages_and_checkpoint_slots(
    make_release_case, attention, mode, page_size, cache_case
):
    context, req, allocator, small_slots, big_slots, expected_indexes = make_release_case(
        attention, mode, page_size, cache_case
    )
    cache = context.radix_cache
    released = []
    context.free_a_req_mem(released, req)
    allocator.free(torch.cat(released))
    # Reusing the request row must not change cached values, including pure_gpu release views.
    context.req_manager.req_to_token_indexs.fill_(-99)
    assert req.cur_kv_len == req.hold_kv_len == req.shm_req.shm_cur_kv_len == 0
    assert req.token_index.token_index_len == req.token_index.gpu_index_submitted_len == 0
    assert req.shared_kv_node is None and req.tail_small_page_buffer_id is None
    assert not req.hybrid_len_to_big_page_id
    cache_len = expected_indexes.numel()
    assert cache.get_refed_tokens_num() == 0
    assert cache.root_node.ref_counter == 1
    assert cache.get_tree_total_tokens_num() == cache_len
    assert allocator.can_use_mem_size == allocator.size - cache_len
    assert big_slots.get_used_cache_num() == (1 if attention != "full" and cache_len > 0 else 0)
    assert small_slots.get_used_cache_num() == (1 if attention == "hybrid_small" else 0)

    key = torch.arange(cache_len, dtype=torch.int64)
    if attention == "full":
        node, matched, indexes = cache.match_prefix(key, update_refs=True)
    else:
        node, matched, indexes = cache.match_prefix(key, block_hashs=[1, 2, 3][: cache_len // 4], update_refs=True)
    assert matched == cache_len
    if cache_len:
        torch.testing.assert_close(indexes, expected_indexes, check_dtype=False)
        cache.dec_node_ref_counter(node)
    else:
        assert node is None and indexes is None

    free_indexes = allocator.alloc(allocator.can_use_mem_size)
    assert len(set(free_indexes.tolist())) == free_indexes.numel()
    assert set(free_indexes.tolist()).isdisjoint(expected_indexes.tolist())
    allocator.free(free_indexes)
    # Real eviction must return every physical KV page and both kinds of checkpoint slot exactly once.
    cache.free_radix_cache_to_get_enough_token(allocator.size)
    assert cache.get_tree_total_tokens_num() == cache.get_refed_tokens_num() == 0
    assert allocator.can_use_mem_size == allocator.size
    assert sorted(allocator.alloc(allocator.size).tolist()) == list(range(allocator.size))
    for slots in (small_slots, big_slots):
        assert slots.get_used_cache_num() == 0
        assert sorted(slots.free_list) == list(range(slots.size))


@pytest.mark.parametrize(
    "checkpoint_lens,computed_len,target_len,inserted_len",
    [
        ([8, 16], 20, 24, 16),
        ([8, 16, 24], 20, 24, 16),
        ([8, 16], 16, 24, 16),
        ([8], 7, 24, 0),
        ([], 7, 24, 0),
        ([8, 16], 20, 12, 8),
        ([8, 16, 24], 24, 24, 24),
        ([8], 20, 12, 8),
    ],
    ids=[
        "multiple",
        "future",
        "at_boundary",
        "future_only",
        "none",
        "target_limit",
        "final_big",
        "decode_after_target",
    ],
)
@pytest.mark.parametrize("mode", ["pin_mem_cache", "pin_mem_page_cache", "pure_gpu"])
@pytest.mark.parametrize("cache_case", ["uncached", "duplicate", "shared"])
def test_partial_prefill_validates_checkpoints_and_caches_completed_prefix(
    make_release_case, monkeypatch, checkpoint_lens, computed_len, target_len, inserted_len, mode, cache_case
):
    context, req, allocator, small_slots, big_slots, first_page_indexes = make_release_case(
        "hybrid_partial", mode, 4, cache_case
    )
    cache = context.radix_cache
    shared_len = 8 if cache_case == "shared" else 0
    req.cur_kv_len = max(computed_len, shared_len)
    req.hybrid_cache_len = target_len
    req.token_index.append_token_indexes(allocator.alloc(8))
    req.hold_kv_len = 24
    for length in list(req.hybrid_len_to_big_page_id):
        if length not in checkpoint_lens:
            big_slots.free_state_cache([req.hybrid_len_to_big_page_id.pop(length)])
    for length in checkpoint_lens:
        if length > shared_len and length not in req.hybrid_len_to_big_page_id:
            req.hybrid_len_to_big_page_id[length] = big_slots.alloc_one_state_cache()

    cache_len = max(inserted_len, 8 if cache_case != "uncached" else 0)
    expected_indexes = req.token_index.get_cpu_token_indexes(0, cache_len)
    if cache_case != "uncached":
        expected_indexes[:8] = first_page_indexes
    checkpoint = req.hybrid_len_to_big_page_id.get(inserted_len)
    insert = Mock(wraps=cache.insert)
    monkeypatch.setattr(cache, "insert", insert)
    released = []
    no_new_big_page = inserted_len <= shared_len
    out_of_range_checkpoint = any(length > max(shared_len, inserted_len) for length in req.hybrid_len_to_big_page_id)
    if req.hybrid_len_to_big_page_id and (no_new_big_page or out_of_range_checkpoint):
        # 无新大页时持有 checkpoint，或 checkpoint 超出插入前缀，均违反请求状态约束。
        checkpoints = dict(req.hybrid_len_to_big_page_id)
        available_tokens = allocator.can_use_mem_size
        cached_tokens = cache.get_tree_total_tokens_num()
        with pytest.raises(AssertionError):
            context.free_a_req_mem(released, req)
        if no_new_big_page:
            insert.assert_not_called()
            assert released == []
            assert req.cur_kv_len == max(computed_len, shared_len)
            assert req.hybrid_len_to_big_page_id == checkpoints
            assert cache.get_tree_total_tokens_num() == cached_tokens
        else:
            assert insert.call_count == 1
            assert len(released) == 1
            assert req.cur_kv_len == inserted_len
            # insert 可能已接管有效 checkpoint，未消费的条目仍由请求持有。
            assert set(req.hybrid_len_to_big_page_id.items()) <= set(checkpoints.items())
            assert cache.get_tree_total_tokens_num() <= inserted_len
        assert req.token_index.token_index_len == req.hold_kv_len == 24
        assert allocator.can_use_mem_size == available_tokens
        assert cache.get_refed_tokens_num() == shared_len
        assert big_slots.get_used_cache_num() == cache.get_tree_total_tokens_num() // 8 + len(
            req.hybrid_len_to_big_page_id
        )
        return
    context.free_a_req_mem(released, req)
    allocator.free(torch.cat(released))
    context.req_manager.req_to_token_indexs.fill_(-99)

    assert insert.call_count == 1
    assert insert.call_args.args[0].numel() == max(shared_len, inserted_len)
    assert req.cur_kv_len == req.hold_kv_len == req.token_index.token_index_len == 0
    assert req.shared_kv_node is None and not req.hybrid_len_to_big_page_id
    assert cache.get_tree_total_tokens_num() == cache_len
    assert cache.get_refed_tokens_num() == 0
    assert allocator.can_use_mem_size == allocator.size - cache_len
    assert small_slots.get_used_cache_num() == 0
    assert big_slots.get_used_cache_num() == cache_len // 8
    node, matched, indexes = cache.match_prefix(
        torch.arange(24, dtype=torch.int64), block_hashs=[1, 2, 3, 4, 5, 6], update_refs=True
    )
    assert matched == cache_len
    if cache_len:
        torch.testing.assert_close(indexes, expected_indexes, check_dtype=False)
        if cache_case == "uncached" or cache_len > 8:
            assert node.big_page_buffer_idx == checkpoint
        cache.dec_node_ref_counter(node)
    else:
        assert node is None and indexes is None

    cache.free_radix_cache_to_get_enough_token(allocator.size)
    assert allocator.can_use_mem_size == allocator.size
    assert sorted(allocator.alloc(allocator.size).tolist()) == list(range(allocator.size))
    assert big_slots.get_used_cache_num() == 0
    assert sorted(big_slots.free_list) == list(range(big_slots.size))


@pytest.mark.parametrize("mode", ["pin_mem_cache", "pin_mem_page_cache", "pure_gpu"])
@pytest.mark.parametrize("reserved_len", [0, 8])
def test_hybrid_release_before_first_forward_handles_reserved_capacity(
    make_release_case, monkeypatch, mode, reserved_len
):
    context, req, allocator, small_slots, big_slots, _ = make_release_case("hybrid_partial", mode, 4, "uncached")
    allocator.free(req.token_index.get_free_token_indexes(0, req.hold_kv_len))
    req.token_index.reset()
    big_slots.free_state_cache(list(req.hybrid_len_to_big_page_id.values()))
    req.hybrid_len_to_big_page_id.clear()
    req.cur_kv_len = 0
    req.hold_kv_len = reserved_len
    if reserved_len:
        req.token_index.append_token_indexes(allocator.alloc(reserved_len))
    insert = Mock(wraps=context.radix_cache.insert)
    monkeypatch.setattr(context.radix_cache, "insert", insert)

    released = []
    context.free_a_req_mem(released, req)
    if reserved_len:
        allocator.free(torch.cat(released))
        assert insert.call_count == 1
        assert insert.call_args.args[0].numel() == insert.call_args.args[1].numel() == 0
    else:
        insert.assert_not_called()
        assert released == []
    assert context.radix_cache.get_tree_total_tokens_num() == context.radix_cache.get_refed_tokens_num() == 0
    assert context.radix_cache.root_node.ref_counter == 1
    assert req.cur_kv_len == req.hold_kv_len == req.token_index.token_index_len == 0
    assert allocator.can_use_mem_size == allocator.size
    assert sorted(allocator.alloc(allocator.size).tolist()) == list(range(allocator.size))
    assert small_slots.get_used_cache_num() == big_slots.get_used_cache_num() == 0


@pytest.mark.parametrize("mode", ["pin_mem_cache", "pin_mem_page_cache", "pure_gpu"])
@pytest.mark.parametrize("shared_prefix_len", [0, 8])
@pytest.mark.parametrize("available_tokens", [3, 4, 5])
def test_hybrid_small_page_match_preserves_tail_capacity_and_checkpoint_order(
    make_release_case, monkeypatch, mode, shared_prefix_len, available_tokens
):
    context, req, allocator, small_slots, big_slots, _ = make_release_case("hybrid_small", mode, 4, "uncached")
    matched_len = shared_prefix_len + 4
    if shared_prefix_len == 0:
        big_slots.free_state_cache(list(req.hybrid_len_to_big_page_id.values()))
        req.hybrid_len_to_big_page_id.clear()
    req.cur_kv_len = req.hybrid_cache_len = matched_len
    small_checkpoint = req.tail_small_page_buffer_id
    cached_indexes = req.token_index.get_cpu_token_indexes(0, matched_len)
    released = []
    context.free_a_req_mem(released, req)
    allocator.free(torch.cat(released))
    free_tokens_before_match = allocator.can_use_mem_size

    req.args = SimpleNamespace(
        page_size=4,
        linear_att_hash_page_size=4,
        linear_att_page_block_num=2,
        max_req_total_len=32,
    )
    req.sampling_param = SimpleNamespace(disable_prompt_cache=False)
    req.get_cur_total_len = lambda: matched_len + 1
    req.shm_req.shm_prompt_ids = SimpleNamespace(arr=np.arange(matched_len + 1))
    block_hashs = list(range(1, matched_len // 4 + 1))
    req.shm_req.hybrid_token_hash_list = SimpleNamespace(size=len(block_hashs), get_all=lambda: block_hashs)
    req.shm_req.prompt_cache_len = 0
    monkeypatch.setattr(context, "get_can_alloc_token_num", lambda: available_tokens)

    operations = []
    copy_kv = Mock(side_effect=lambda *args: operations.append("copy_kv"))
    context.radix_cache.mem_manager.operator = SimpleNamespace(copy_mem_to_mem=copy_kv)
    copy_logprobs = Mock(side_effect=lambda **kwargs: operations.append("copy_logprobs"))
    req.prompt_selected_logprobs = SimpleNamespace(copy_capture_slots_if_needed=copy_logprobs)

    def restore_small_page(req):
        assert req.shared_kv_node.small_page_buffer_idx == small_checkpoint
        assert req.shared_kv_node.ref_counter > 0
        operations.append("restore_small")

    context.req_manager.restore_small_page_state = restore_small_page
    context.req_manager.restore_big_page_state = Mock()
    context.req_manager.init_hybrid_attention_state = Mock()
    deref = context.radix_cache.deref_to_first_big_page_node

    def release_small_page_reference(node):
        operations.append("deref")
        return deref(node)

    monkeypatch.setattr(context.radix_cache, "deref_to_first_big_page_node", release_small_page_reference)
    req._hybrid_match_radix_cache()

    rebuild_tail = available_tokens > 4
    expected_len = matched_len if rebuild_tail else shared_prefix_len
    expected_indexes = cached_indexes[:shared_prefix_len]
    if rebuild_tail:
        source_indexes, private_indexes = copy_kv.call_args.args
        torch.testing.assert_close(source_indexes, cached_indexes[shared_prefix_len:])
        assert private_indexes.numel() == 4
        assert not torch.isin(private_indexes, cached_indexes).any()
        copy_logprobs.assert_called_once()
        torch.testing.assert_close(copy_logprobs.call_args.kwargs["source_indexes"], source_indexes)
        torch.testing.assert_close(copy_logprobs.call_args.kwargs["destination_indexes"], private_indexes)
        expected_indexes = torch.cat([expected_indexes, private_indexes])
        assert operations == ["copy_kv", "copy_logprobs", "restore_small", "deref"]
    else:
        copy_kv.assert_not_called()
        copy_logprobs.assert_not_called()
        assert operations == ["deref"]

    assert req.cur_kv_len == req.hold_kv_len == req.shm_req.shm_cur_kv_len == expected_len
    assert req.shm_req.prompt_cache_len == req.token_index.token_index_len == expected_len
    torch.testing.assert_close(req.token_index.get_cpu_token_indexes(0, expected_len), expected_indexes)
    assert allocator.can_use_mem_size == free_tokens_before_match - (4 if rebuild_tail else 0)
    assert context.radix_cache.get_refed_tokens_num() == shared_prefix_len
    if shared_prefix_len:
        assert req.shared_kv_node.is_big_page_node()
        assert req.shared_kv_node.node_prefix_total_len == shared_prefix_len
    else:
        assert req.shared_kv_node is None
    if not rebuild_tail and shared_prefix_len:
        context.req_manager.restore_big_page_state.assert_called_once_with(
            big_page_buffer_idx=req.shared_kv_node.big_page_buffer_idx, req=req
        )
    else:
        context.req_manager.restore_big_page_state.assert_not_called()
    if expected_len == 0:
        context.req_manager.init_hybrid_attention_state.assert_called_once_with(req=req)
    else:
        context.req_manager.init_hybrid_attention_state.assert_not_called()

    released = []
    context.free_a_req_mem(released, req)
    if released:
        allocator.free(torch.cat(released))
    context.radix_cache.clear_tree_nodes()
    assert allocator.can_use_mem_size == allocator.size
    assert small_slots.get_used_cache_num() == big_slots.get_used_cache_num() == 0


@pytest.mark.parametrize("mode", ["pin_mem_cache", "pin_mem_page_cache", "pure_gpu"])
@pytest.mark.parametrize("new_tokens", [0, 1])
@pytest.mark.parametrize("evict_checkpoint", [False, True])
def test_hybrid_release_skips_insert_for_shared_small_prefix(
    make_release_case, monkeypatch, mode, new_tokens, evict_checkpoint
):
    context, req, allocator, small_slots, big_slots, expected_indexes = make_release_case(
        "hybrid_small", mode, 4, "uncached"
    )
    # Use the real match path with big pages disabled, which retains a shared small node.
    big_slots.free_state_cache(list(req.hybrid_len_to_big_page_id.values()))
    req.hybrid_len_to_big_page_id.clear()
    context.args.linear_att_page_block_num = 10000000
    context.radix_cache = hybrid_att_radix_cache.HybridAttPagedRadixCache(
        64,
        0,
        hash_page_size=4,
        big_page_num=context.args.linear_att_page_block_num,
        kv_cache_mem_manager=context.radix_cache.mem_manager,
        small_page_buffers=small_slots,
    )
    req.args = SimpleNamespace(
        req_token_index_mode=mode,
        page_size=4,
        linear_att_hash_page_size=4,
        linear_att_page_block_num=context.args.linear_att_page_block_num,
        max_req_total_len=32,
    )
    req.sampling_param = SimpleNamespace(disable_prompt_cache=False)
    req.get_cur_total_len = lambda: 13
    req.shm_req.shm_prompt_ids = SimpleNamespace(arr=np.arange(13))
    req.shm_req.hybrid_token_hash_list = SimpleNamespace(size=3, get_all=lambda: [1, 2, 3])
    restored_checkpoints = []
    context.req_manager.restore_small_page_state = lambda req: restored_checkpoints.append(
        req.shared_kv_node.small_page_buffer_idx
    )
    released = []
    context.free_a_req_mem(released, req)
    allocator.free(torch.cat(released))
    cache = context.radix_cache
    req._hybrid_match_radix_cache()
    node = req.shared_kv_node
    shared_len = req.cur_kv_len
    checkpoint = node.small_page_buffer_idx
    assert shared_len == 12 and checkpoint is not None
    assert not node.is_big_page_node()
    assert restored_checkpoints == [checkpoint]
    reserved_checkpoints = []
    if evict_checkpoint:
        # Exhaust the slot pool, then recycle the checkpoint while its node is still referenced.
        reserved_checkpoints = small_slots.alloc_state_cache(small_slots.get_free_cache_num())
        cache.free_one_small_page_buffer()
        assert node.ref_counter > 0 and node.small_page_buffer_idx is None
        reused_checkpoint = small_slots.alloc_one_state_cache()
        assert reused_checkpoint == checkpoint
        reserved_checkpoints.append(reused_checkpoint)
        small_slots.states[reused_checkpoint].fill_(42)
    req.cur_kv_len = shared_len + new_tokens
    req.hold_kv_len = 16
    req.token_index.append_token_indexes(allocator.alloc(req.hold_kv_len - shared_len))
    insert = Mock(wraps=cache.insert)
    monkeypatch.setattr(cache, "insert", insert)

    released = []
    context.free_a_req_mem(released, req)
    allocator.free(torch.cat(released))
    context.req_manager.req_to_token_indexs.fill_(-99)

    insert.assert_not_called()
    assert req.shared_kv_node is None
    assert req.cur_kv_len == req.hold_kv_len == req.token_index.token_index_len == 0
    assert cache.get_tree_total_tokens_num() == shared_len
    assert cache.get_refed_tokens_num() == 0
    assert allocator.can_use_mem_size == allocator.size - shared_len
    assert big_slots.get_used_cache_num() == 0
    assert small_slots.get_used_cache_num() == (small_slots.size if evict_checkpoint else 1)
    assert node.small_page_buffer_idx == (None if evict_checkpoint else checkpoint)
    if evict_checkpoint:
        assert small_slots.states[reused_checkpoint].item() == 42
    node, matched, indexes = cache.match_prefix(
        torch.arange(12, dtype=torch.int64), block_hashs=[1, 2, 3], update_refs=True
    )
    if evict_checkpoint:
        assert node is None and indexes is None and matched == 0
        small_slots.free_state_cache(reserved_checkpoints)
    else:
        assert matched == shared_len
        torch.testing.assert_close(indexes, expected_indexes, check_dtype=False)
        cache.dec_node_ref_counter(node)
    cache.free_radix_cache_to_get_enough_token(allocator.size)
    assert allocator.can_use_mem_size == allocator.size
    assert sorted(allocator.alloc(allocator.size).tolist()) == list(range(allocator.size))
    for slots in (small_slots, big_slots):
        assert slots.get_used_cache_num() == 0
        assert sorted(slots.free_list) == list(range(slots.size))


@pytest.mark.parametrize("prefix_len", [4, 12])
def test_hybrid_new_small_prefix_still_requires_checkpoint(make_release_case, prefix_len):
    context, req, _, _, big_slots, _ = make_release_case("hybrid_partial", "pin_mem_cache", 4, "uncached")
    indexes = req.token_index.get_cpu_token_indexes(0, prefix_len)
    big_checkpoints = req.hybrid_len_to_big_page_id if prefix_len == 12 else SortedDict()
    original_checkpoints = dict(big_checkpoints)
    with pytest.raises(AssertionError):
        context.radix_cache.insert(
            torch.arange(prefix_len, dtype=torch.int64),
            indexes,
            block_hashs=[1, 2, 3][: prefix_len // 4],
            block_state_idxs=[None] * (prefix_len // 4),
            len_to_big_page_id=big_checkpoints,
        )
    assert context.radix_cache.get_tree_total_tokens_num() == 0
    assert context.radix_cache.root_node.children == {}
    assert big_checkpoints == original_checkpoints
    assert big_slots.get_used_cache_num() == 1


@pytest.mark.parametrize("attention", ["hybrid_small", "hybrid_partial"])
@pytest.mark.parametrize("failed_range", ["tail", "cached_prefix"])
def test_hybrid_snapshot_failure_keeps_checkpoint_owned_by_request(
    make_release_case, monkeypatch, attention, failed_range
):
    context, req, allocator, small_slots, big_slots, _ = make_release_case(attention, "pin_mem_cache", 4, "uncached")
    original_kv_len = req.cur_kv_len
    cache_len = 12 if attention == "hybrid_small" else 8
    checkpoint = req.tail_small_page_buffer_id
    big_checkpoints = dict(req.hybrid_len_to_big_page_id)
    get_indexes = req.token_index.get_cpu_token_indexes

    def failed_snapshot(start, end):
        if (start == 0) == (failed_range == "cached_prefix"):
            raise RuntimeError("snapshot failed")
        return get_indexes(start, end)

    with monkeypatch.context() as failure:
        failure.setattr(req.token_index, "get_cpu_token_indexes", failed_snapshot)
        with pytest.raises(RuntimeError, match="snapshot failed"):
            context.free_a_req_mem([], req)
    assert req.tail_small_page_buffer_id == checkpoint
    assert req.hybrid_len_to_big_page_id == big_checkpoints
    assert req.cur_kv_len == (original_kv_len if failed_range == "tail" else cache_len)
    assert small_slots.get_used_cache_num() == (1 if attention == "hybrid_small" else 0)
    assert big_slots.get_used_cache_num() == len(big_checkpoints)
    assert context.radix_cache.get_tree_total_tokens_num() == 0
    released = []
    context.free_a_req_mem(released, req)
    allocator.free(torch.cat(released))
    context.radix_cache.free_radix_cache_to_get_enough_token(allocator.size)
    assert allocator.can_use_mem_size == allocator.size
    assert small_slots.get_used_cache_num() == big_slots.get_used_cache_num() == 0
