import uuid
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sortedcontainers import SortedDict

from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
from lightllm.server.router.dynamic_prompt.hybrid_att_radix_cache import HybridAttPagedRadixCache
from lightllm.server.router.dynamic_prompt.tool_call_cache import contains_tool_call
from lightllm.utils import shm_utils


@pytest.fixture(autouse=True)
def service_name(monkeypatch):
    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: f"tool_cache_{uuid.uuid4().hex}")


def insert(tree, tokens):
    return tree.insert(torch.tensor(tokens, dtype=torch.int64))[1]


def cache():
    tree = RadixCache(40, 0)
    tree.enable_tool_call_retention(10, 0.5)
    tree.tool_call_retention.advance_clock(100)
    return tree


def test_prefers_tool_prefix_then_restores_original_recency():
    tree = cache()
    tool = insert(tree, [1, 2, 3])
    ordinary = insert(tree, [4, 5, 6])
    tree.tool_call_retention.protect(tool)
    assert tree.evict_tree_set[0] is ordinary
    tree.tool_call_retention.advance_clock(110)
    assert tree.evict_tree_set[0] is tool
    assert tree.tool_call_retention.protected_tokens == 0
    tree.evict_tree_set._check()


def test_pressure_can_evict_every_protected_token():
    tree = cache()
    for tokens in ([1, 2, 3], [4, 5, 6]):
        tree.tool_call_retention.protect(insert(tree, tokens))
    evicted = []
    tree.evict(6, lambda values: evicted.extend(values.tolist()))
    assert len(evicted) == 6
    assert tree.get_tree_total_tokens_num() == 0
    assert tree.tool_call_retention.protected_tokens == 0
    assert not tree.tool_call_retention._leases


def test_references_are_stronger_than_retention():
    tree = cache()
    insert(tree, [1, 2, 3])
    active, _, _ = tree.match_prefix(torch.tensor([1, 2, 3]), update_refs=True)
    tool = insert(tree, [4, 5, 6])
    tree.tool_call_retention.protect(tool)
    tree.evict(3, lambda _: None)
    assert tree.get_tree_total_tokens_num() == 3
    assert active.ref_counter == 1
    tree.dec_node_ref_counter(active)


def test_budget_and_oversized_prefix():
    tree = cache()
    first = insert(tree, list(range(10)))
    second = insert(tree, list(range(30, 45)))
    tree.tool_call_retention.protect(first)
    tree.tool_call_retention.protect(second)
    assert not first.cache_protected
    assert second.cache_protected
    assert tree.tool_call_retention.protected_tokens == 20
    oversized = insert(tree, list(range(100, 121)))
    tree.tool_call_retention.protect(oversized)
    assert not oversized.cache_protected
    assert oversized.parent.cache_protected
    assert tree.tool_call_retention.protected_tokens == 20


@pytest.mark.parametrize("page_size", [1, 2, 4])
def test_oversized_context_keeps_a_page_aligned_partial_prefix(page_size):
    tree = RadixCache(20, 0, page_size=page_size)
    tree.enable_tool_call_retention(10, 0.25)
    original = list(range(12))
    tail = insert(tree, original)
    tree.tool_call_retention.protect(tail)
    kept = 5 // page_size * page_size
    assert tree.tool_call_retention.protected_tokens == kept
    insert(tree, list(range(30, 38)))
    tree.evict(12, lambda _: None)
    _, matched, _ = tree.match_prefix(torch.tensor(original), update_refs=False)
    assert matched == kept
    tree.evict_tree_set._check()


def test_partial_prefix_split_preserves_active_request_references():
    tree = RadixCache(20, 0)
    tree.enable_tool_call_retention(10, 0.25)
    original = torch.arange(12)
    tree.insert(original)
    active, _, _ = tree.match_prefix(original, update_refs=True)
    tree.tool_call_retention.protect(active)
    assert active.node_prefix_total_len == 12
    assert active.ref_counter == active.parent.ref_counter == 1
    assert tree.get_refed_tokens_num() == 12
    tree.dec_node_ref_counter(active)
    assert tree.get_refed_tokens_num() == 0
    tree.evict(7, lambda _: None)
    _, matched, values = tree.match_prefix(original, update_refs=False)
    assert matched == 5
    assert values.tolist() == list(range(5))


def test_oversized_hybrid_prefix_uses_existing_resumable_checkpoint():
    buffers = SimpleNamespace(free_state_cache=lambda *args, **kwargs: None)
    tree = HybridAttPagedRadixCache(20, 0, 2, 2, SimpleNamespace(big_page_buffers=buffers), buffers)
    tree.enable_tool_call_retention(10, 0.25)
    _, tail = tree.insert(
        torch.arange(12),
        block_hashs=[11, 22, 33, 44, 55, 66],
        block_state_idxs=[None] * 6,
        len_to_big_page_id=SortedDict({4: 100, 8: 101, 12: 102}),
    )
    tree.tool_call_retention.protect(tail)
    assert not tail.cache_protected
    assert not tail.parent.cache_protected
    assert tail.parent.parent.cache_protected
    assert tail.parent.parent.big_page_buffer_idx == 100
    assert tree.tool_call_retention.protected_tokens == 4


def test_hybrid_does_not_prefer_a_prefix_without_recurrent_state():
    buffers = SimpleNamespace(free_state_cache=lambda *args, **kwargs: None)
    tree = HybridAttPagedRadixCache(20, 0, 2, 1000, SimpleNamespace(big_page_buffers=buffers), buffers)
    tree.enable_tool_call_retention(10, 0.25)
    _, tail = tree.insert(torch.arange(12), block_hashs=[11, 22, 33, 44, 55, 66], block_state_idxs=[None] * 5 + [10])
    tree.tool_call_retention.protect(tail)
    assert tree.tool_call_retention.protected_tokens == 0


def test_evicted_recurrent_state_falls_back_to_a_valid_older_checkpoint():
    freed = []
    buffers = SimpleNamespace(
        free_state_cache=lambda indexes=None, free_indexes=None: freed.extend(indexes or free_indexes),
        get_free_cache_num=lambda: 0,
    )
    tree = HybridAttPagedRadixCache(20, 0, 2, 1000, SimpleNamespace(big_page_buffers=buffers), buffers)
    tree.enable_tool_call_retention(10, 0.5)
    _, older = tree.insert(torch.arange(2), block_hashs=[11], block_state_idxs=[10])
    _, newer = tree.insert(torch.arange(4), block_hashs=[11, 22], block_state_idxs=[None, 20])
    tree.tool_call_retention.protect(newer)
    matched, _, _ = tree.match_prefix(torch.arange(2), block_hashs=[11], update_refs=True)
    tree.dec_node_ref_counter(matched)
    tree.free_one_small_page_buffer()
    assert freed == [20]
    assert not newer.cache_protected
    assert older.cache_protected
    assert tree.tool_call_retention.protected_tokens == 2
    tree.free_one_small_page_buffer()
    assert tree.tool_call_retention.protected_tokens == 0
    tree._evict_tree_set._check()


@pytest.mark.parametrize("split_by_insert", [False, True])
def test_split_preserves_lease_deadline_and_token_accounting(split_by_insert):
    tree = cache()
    tail = insert(tree, [1, 2, 3, 4, 5, 6])
    retention = tree.tool_call_retention
    retention.protect(tail)
    if split_by_insert:
        insert(tree, [1, 2, 3, 8, 9])
    else:
        tree.match_prefix(torch.tensor([1, 2, 3]), update_refs=False)
    assert tail.cache_protected and tail.parent.cache_protected
    assert retention.protected_tokens == 6
    assert len(retention._leases) == 2
    tree.merge_unreferenced_nodes()
    assert retention.protected_tokens == 6
    tree.evict_tree_set._check()
    retention.advance_clock(110)
    tree.merge_unreferenced_nodes()
    assert retention.protected_tokens == 0
    tree.evict_tree_set._check()


def test_shared_prefix_counted_once_and_renewal_is_bounded():
    tree = cache()
    left = insert(tree, [1, 2, 3, 4])
    right = insert(tree, [1, 2, 5, 6])
    retention = tree.tool_call_retention
    retention.protect(left)
    retention.protect(right)
    assert retention.protected_tokens == 6
    for _ in range(100):
        retention.protect(right)
    assert len(retention._expiry) == len(retention._leases) == 3
    retention.advance_clock(109)
    retention.protect(right)
    retention.advance_clock(110)
    assert not left.cache_protected
    assert right.cache_protected and right.parent.cache_protected
    assert retention.protected_tokens == 4


def test_budget_counts_structurally_retained_ancestors():
    tree = cache()
    insert(tree, list(range(10)))
    old_tail = insert(tree, list(range(15)))
    tree.tool_call_retention.protect(old_tail)
    newer = insert(tree, list(range(30, 40)))
    tree.tool_call_retention.protect(newer)
    assert newer.cache_protected
    assert not old_tail.cache_protected
    assert old_tail.parent.cache_protected
    assert tree.tool_call_retention.protected_tokens == 20


def test_final_answer_revokes_stale_tool_prefix_but_preserves_other_branches():
    tree = cache()
    left = insert(tree, [1, 2, 3, 4])
    right = insert(tree, [1, 2, 5, 6])
    retention = tree.tool_call_retention
    retention.protect(left)
    retention.protect(right)
    final = insert(tree, [1, 2, 3, 4, 7, 8])
    retention.complete(final)
    assert not left.cache_protected
    assert right.cache_protected
    assert right.parent.cache_protected
    assert retention.protected_tokens == 4
    tree.evict_tree_set._check()
    retention.complete(right)
    assert retention.protected_tokens == 0
    tree.evict_tree_set._check()


def test_hybrid_retains_both_kv_and_state_without_pinning():
    freed = []
    buffers = SimpleNamespace(
        free_state_cache=lambda indexes=None, free_indexes=None: freed.extend(indexes or free_indexes),
        get_free_cache_num=lambda: 0,
    )
    tree = HybridAttPagedRadixCache(40, 0, 2, 2, SimpleNamespace(big_page_buffers=buffers), buffers)
    tree.enable_tool_call_retention(10, 0.5)
    _, tool = tree.insert(torch.tensor([1, 2]), block_hashs=[11], block_state_idxs=[10])
    _, ordinary = tree.insert(torch.tensor([3, 4]), block_hashs=[12], block_state_idxs=[20])
    tree.tool_call_retention.protect(tool)
    assert tree._evict_tree_set[0] is ordinary
    tree.free_one_small_page_buffer()
    assert freed == [20]
    # A node with no reusable recurrent state must still be reclaimed first.
    assert tree._evict_tree_set[0] is ordinary
    tree._evict(4, lambda values, state: None)
    assert tree.tool_call_retention.protected_tokens == 0


@pytest.mark.parametrize(
    "tokens, markers, expected",
    [
        ([4, 5, 6], [[5, 6]], True),
        ([4, 5], [[5, 6]], False),
        ([], [[5]], False),
        ([4], [[]], False),
        (np.array([1, 2, 3]), [[2]], True),
    ],
)
def test_generated_marker_detection(tokens, markers, expected):
    assert contains_tool_call(tokens, markers) is expected


def test_compact_markers_round_trip_and_do_not_become_stop_sequences():
    from lightllm.server.core.objs.sampling_params import SamplingParams, ToolCallCacheMarkers
    import ctypes

    assert ctypes.sizeof(ToolCallCacheMarkers) < 1024
    params = SamplingParams()
    params.init(None, tool_call_cache_markers=[[8, 9], [10]], stop_sequences=[[11]])
    copy = SamplingParams()
    copy.init(None, **params.to_origin_dict())
    assert copy.tool_call_cache_markers.to_list() == [[8, 9], [10]]
    assert copy.stop_sequences.to_token_ids_list() == [[11]]
    copy.init(None)
    assert copy.tool_call_cache_markers.to_list() == []


@pytest.mark.parametrize(
    "parser", ["qwen25", "qwen3_coder", "kimi_k2", "deepseekv3", "deepseekv32", "glm47", "llama3", "mistral"]
)
def test_supported_parser_markers(parser):
    from lightllm.server.function_call_parser import FunctionCallParser

    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [ord(c) for c in text])
    markers = FunctionCallParser.cache_marker_token_ids(parser, tokenizer)
    assert markers
    if parser == "qwen3_coder":
        assert list(map(ord, "<function=")) in markers
    if parser == "kimi_k2":
        assert list(map(ord, "<|tool_call_begin|>")) in markers


@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_scheduling_clock_is_packed_once_and_shared_across_ranks(monkeypatch, ready, enabled):
    from lightllm.server.router.model_infer.mode_backend import base_backend

    transmitted = []
    read = []
    ranks = []
    clocks = [[], []]
    for index in range(2):
        backend = base_backend.ModeBackend.__new__(base_backend.ModeBackend)
        backend.is_master_in_node = index == 0
        backend.enable_tool_call_retention = enabled
        backend.args = SimpleNamespace(node_rank=0)
        backend.node_world_size = 2
        backend.node_nccl_group = None
        backend.shm_reqs_io_buffer = SimpleNamespace(is_ready=lambda: ready)
        backend.node_broadcast_tensor = torch.tensor([0], dtype=torch.int64 if enabled else torch.int32)
        backend.is_pd_mode = False
        backend.radix_cache = SimpleNamespace(tool_call_retention=SimpleNamespace(advance_clock=clocks[index].append))
        backend._read_reqs_buffer_and_init_reqs = lambda index=index: read.append(index)
        ranks.append(backend)

    def broadcast(tensor, **kwargs):
        if tensor is ranks[0].node_broadcast_tensor:
            transmitted.append(tensor.clone())
        else:
            tensor.copy_(transmitted[-1])

    monkeypatch.setattr(base_backend, "broadcast", broadcast)
    monkeypatch.setattr(base_backend.time, "monotonic_ns", lambda: 123456789000)
    ranks[0]._try_read_new_reqs_normal()
    monkeypatch.setattr(base_backend.time, "monotonic_ns", lambda: 987654321000)
    ranks[1]._try_read_new_reqs_normal()
    assert read == ([0, 1] if ready else [])
    assert clocks == ([[123.456789], [123.456789]] if enabled else [[], []])
    assert transmitted[0].numel() == 1


@pytest.mark.parametrize("all_ready", [False, True])
def test_multinode_expiry_uses_rank_zero_clock_and_masks_readiness(monkeypatch, all_ready):
    from lightllm.server.router.model_infer.mode_backend import base_backend

    backend = base_backend.ModeBackend.__new__(base_backend.ModeBackend)
    backend.enable_tool_call_retention = True
    backend.shm_reqs_io_buffer = SimpleNamespace(is_ready=lambda: all_ready)
    backend.multinode_tp_gather_item_tensor = torch.zeros(1, dtype=torch.int64)
    backend.multinode_tp_all_gather_tensor = torch.zeros(2, dtype=torch.int64)
    backend.multinode_tp_nccl_group = None
    backend.is_pd_mode = False
    clocks = []
    initialized = []
    backend.radix_cache = SimpleNamespace(tool_call_retention=SimpleNamespace(advance_clock=clocks.append))
    backend._read_reqs_buffer_and_init_reqs = lambda: initialized.append(True)

    def gather(output, value, **kwargs):
        # Rank zero is ready, but the peer may not be. Their local clocks differ.
        output.copy_(torch.tensor([(123456789 << 1) | 1, (987654321 << 1) | int(all_ready)]))

    monkeypatch.setattr(base_backend, "all_gather_into_tensor", gather)
    backend._try_read_new_reqs_multinode_tp()
    assert clocks == [123.456789]
    assert initialized == ([True] if all_ready else [])


@pytest.mark.parametrize(
    "status, enabled, generated, expected",
    [
        (1, True, [8, 9], True),
        (2, True, [8, 9], False),
        (3, True, [8, 9], False),
        (4, True, [8, 9], False),
        (0, True, [8, 9], False),
        (1, False, [8, 9], False),
        (1, True, [7, 9], False),
    ],
)
def test_only_normal_stop_with_generated_tool_signal_receives_lease(status, enabled, generated, expected):
    from lightllm.server.core.objs import FinishStatus
    from lightllm.server.core.objs.sampling_params import ToolCallCacheMarkers
    from lightllm.server.router.model_infer.infer_batch import InferenceContext

    tree = cache()
    node = insert(tree, [1, 2, 3])
    marker = ToolCallCacheMarkers()
    marker.initialize([[8, 9]])
    req = SimpleNamespace(
        finish_status=FinishStatus(status),
        finish_output_len=len(generated),
        sampling_param=SimpleNamespace(
            shm_param=SimpleNamespace(tool_call_cache_markers=marker), disable_prompt_cache=not enabled
        ),
        shm_req=SimpleNamespace(input_len=2),
        # The history contains a tool marker even when generated output does not.
        get_input_token_ids=lambda: np.array([8, 9] + generated),
    )
    InferenceContext(radix_cache=tree)._retain_tool_call_prefix(req, node)
    assert node.cache_protected is expected


def test_post_eos_speculative_tool_marker_does_not_receive_a_preference():
    from lightllm.server.core.objs import FinishStatus
    from lightllm.server.core.objs.sampling_params import ToolCallCacheMarkers
    from lightllm.server.router.model_infer.infer_batch import InferenceContext

    tree = cache()
    node = insert(tree, [1, 2, 3])
    marker = ToolCallCacheMarkers()
    marker.initialize([[8, 9]])
    req = SimpleNamespace(
        finish_status=FinishStatus(FinishStatus.FINISHED_STOP),
        finish_output_len=1,
        sampling_param=SimpleNamespace(
            shm_param=SimpleNamespace(tool_call_cache_markers=marker), disable_prompt_cache=False
        ),
        shm_req=SimpleNamespace(input_len=2),
        get_input_token_ids=lambda: np.array([1, 2, 7, 8, 9]),
    )
    InferenceContext(radix_cache=tree)._retain_tool_call_prefix(req, node)
    assert not node.cache_protected


@pytest.mark.parametrize("is_master", [False, True])
@pytest.mark.parametrize("callback_finishes", [False, True])
def test_later_speculative_packs_keep_kv_updates_but_preserve_first_finish_boundary(is_master, callback_finishes):
    from lightllm.server.core.objs import FinishStatus
    from lightllm.server.router.model_infer.infer_batch import InferReqUpdatePack

    accepted = []
    status = FinishStatus()
    shm = SimpleNamespace(input_len=4)

    def finish(**kwargs):
        status.set_status(FinishStatus.FINISHED_STOP)

    req = SimpleNamespace(
        shm_req=shm,
        finish_status=status,
        finish_output_len=0,
        output_constraint=None,
        set_next_gen_token_id=lambda token, logprob, length, rank: accepted.append((token, length)),
        update_finish_status=(lambda **kwargs: None) if callback_finishes else finish,
    )
    callback = (lambda *args: finish()) if callback_finishes else None
    for length, token in [(2, 7), (3, 8), (4, 9)]:
        InferReqUpdatePack(req, length).handle(token, 0.0, 0, [7], is_master, callback)
    assert accepted == [(7, 2), (8, 3), (9, 4)]
    assert req.finish_output_len == 2
    if is_master:
        assert shm.finish_token_index == 5
        assert shm.shm_cur_output_len == 2


@pytest.mark.parametrize("is_master", [False, True])
@pytest.mark.parametrize("grammar_error", [False, True])
def test_constraint_completion_preserves_the_accepted_cache_boundary(is_master, grammar_error):
    from lightllm.server.core.objs import FinishStatus
    from lightllm.server.router.model_infer.infer_batch import InferReqUpdatePack

    committed = []
    constraint = SimpleNamespace(error=None, is_terminated=lambda: True)

    def commit(token):
        committed.append(token)
        if grammar_error:
            constraint.error = "invalid sampled token"

    constraint.commit = commit
    req = SimpleNamespace(
        shm_req=SimpleNamespace(input_len=4),
        finish_status=FinishStatus(),
        finish_output_len=0,
        output_constraint=constraint,
        set_next_gen_token_id=lambda *args, **kwargs: None,
        update_finish_status=lambda **kwargs: None,
    )
    InferReqUpdatePack(req, 2).handle(7, 0.0, 0, [], is_master)
    InferReqUpdatePack(req, 3).handle(8, 0.0, 0, [], is_master)
    assert committed == [7]
    assert req.finish_output_len == 2
    expected = FinishStatus.FINISHED_ERROR if grammar_error else FinishStatus.FINISHED_STOP
    assert req.finish_status.status == expected


@pytest.mark.parametrize("seed", range(8))
def test_randomized_split_expiry_eviction_and_rank_agreement(seed):
    rng = random.Random(seed)
    ranks = [cache(), cache()]
    for step in range(250):
        tokens = [rng.randrange(4) for _ in range(rng.randrange(1, 15))]
        protect = rng.random() < 0.5
        evict = rng.randrange(8)
        outputs = []
        for tree in ranks:
            retention = tree.tool_call_retention
            retention.advance_clock(100 + step / 4)
            node = insert(tree, tokens)
            if protect:
                retention.protect(node)
            else:
                retention.complete(node)
            shared, _, _ = tree.match_prefix(torch.tensor(tokens[:4]), update_refs=True)
            if shared is not None:
                tree.dec_node_ref_counter(shared)
            tree.merge_unreferenced_nodes()
            values = []
            if tree.get_tree_total_tokens_num() >= evict:
                tree.evict(evict, lambda tensor: values.extend(tensor.tolist()))
            outputs.append(values)
            assert retention.protected_tokens == sum(len(n.token_id_key) for n in retention._leases)
            assert 0 <= retention.protected_tokens <= retention.token_budget
            assert retention.reserved_tokens == sum(n.node_value_len for n in retention._leases if n.ref_counter == 0)
            assert tree.tool_call_reserved_tokens.arr[0] == retention.reserved_tokens
            assert len(retention._expiry) == len(retention._leases)
            for node in retention._leases:
                while node.parent is not None:
                    assert node.cache_protected
                    node = node.parent
            assert set(retention._frontier.values()) == {
                n for n in retention._leases if not any(c.cache_protected for c in n.children.values())
            }
            tree.evict_tree_set._check()
        assert outputs[0] == outputs[1]


def test_budget_preserves_older_partial_prefix_instead_of_dropping_whole_branch():
    tree = cache()
    old = insert(tree, list(range(10)))
    new = insert(tree, list(range(30, 45)))
    tree.tool_call_retention.protect(old)
    tree.tool_call_retention.protect(new)
    assert tree.tool_call_retention.protected_tokens == 20
    assert old.parent.node_prefix_total_len == 5
    assert old.parent.cache_protected
    tree.evict(5, lambda _: None)
    assert tree.match_prefix(torch.arange(10), update_refs=False)[1] == 5
    assert tree.match_prefix(torch.arange(30, 45), update_refs=False)[1] == 15
    tree.evict_tree_set._check()


def test_budget_trims_shared_branch_without_discarding_shared_ancestor():
    tree = RadixCache(20, 0)
    tree.enable_tool_call_retention(10, 0.5)
    left = insert(tree, [1, 2, 3, 4])
    right = insert(tree, [1, 2, 5, 6])
    other = insert(tree, list(range(30, 38)))
    retention = tree.tool_call_retention
    for node in [left, right, other]:
        retention.protect(node)
    assert retention.protected_tokens == 10
    assert not left.cache_protected and not right.cache_protected
    assert right.parent.cache_protected
    assert other.cache_protected
    tree.evict_tree_set._check()


def test_inflight_continuation_releases_waiting_budget_without_unpinning_kv():
    tree = cache()
    active = insert(tree, list(range(10)))
    waiting = insert(tree, list(range(30, 45)))
    retention = tree.tool_call_retention
    retention.protect(active)
    active, matched, _ = tree.match_prefix(torch.arange(10), update_refs=True)
    assert matched == 10
    assert retention.protected_tokens == 0
    assert active.ref_counter == 1
    retention.protect(waiting)
    assert waiting.cache_protected
    assert retention.protected_tokens == 15
    tree.evict(15, lambda _: None)
    assert active.ref_counter == 1
    tree.dec_node_ref_counter(active)
    tree.evict_tree_set._check()


def test_partial_trimming_preserves_original_lease_priority_and_expiry():
    tree = cache()
    old = insert(tree, list(range(10)))
    newer = insert(tree, list(range(30, 45)))
    retention = tree.tool_call_retention
    retention.protect(old)
    original_priority = retention._leases[old][0][:2]
    retention.protect(newer)
    assert retention._leases[old.parent][0][:2] == original_priority
    assert retention.get_stats()["budget_tokens"] == 5
    assert retention._frontier.peekitem(0)[1] is old.parent
    retention.advance_clock(110)
    assert retention.get_stats()["expired_tokens"] == 20
    assert retention.get_stats()["waiting_endpoints"] == 0


def test_hybrid_consume_then_tool_stop_reacquires_only_updated_waiting_prefix():
    buffers = SimpleNamespace(free_state_cache=lambda *args, **kwargs: None)
    tree = HybridAttPagedRadixCache(20, 0, 2, 2, SimpleNamespace(big_page_buffers=buffers), buffers)
    tree.enable_tool_call_retention(10, 0.5)
    _, tool = tree.insert(
        torch.arange(4), block_hashs=[11, 22], block_state_idxs=[None, None], len_to_big_page_id=SortedDict({4: 100})
    )
    retention = tree.tool_call_retention
    retention.protect(tool)
    matched, length, _ = tree.match_prefix(torch.arange(4), block_hashs=[11, 22], update_refs=True)
    assert length == 4 and matched.ref_counter == 1
    assert retention.protected_tokens == 0
    assert retention.get_stats()["consumed_tokens"] == 4
    retention.protect(matched)
    tree.dec_node_ref_counter(matched)
    assert matched.cache_protected and matched.ref_counter == 0
    assert retention.protected_tokens == 4
    tree._evict_tree_set._check()


def test_shared_reservation_excludes_pinned_ancestors_and_renews_without_transient_drops():
    tree = cache()
    left = insert(tree, [1, 2, 3, 4])
    right = insert(tree, [1, 2, 5, 6])
    retention = tree.tool_call_retention
    retention.protect(left)
    retention.protect(right)
    assert retention.reserved_tokens == 6
    prefix, length, _ = tree.match_prefix(torch.tensor([1, 2]), update_refs=True)
    assert length == 2 and retention.protected_tokens == 6
    assert retention.reserved_tokens == 4
    tree.add_node_ref_counter(prefix)
    assert retention.reserved_tokens == 4
    tree.dec_node_ref_counter(prefix)
    assert retention.reserved_tokens == 4
    tree.dec_node_ref_counter(prefix)
    assert retention.reserved_tokens == 6
    published = []
    retention._publish_reserved = published.append
    retention.protect(right)
    assert published == []
    retention.advance_clock(110)
    assert retention.reserved_tokens == retention.protected_tokens == 0


def test_reservation_tracks_protected_split_and_remains_fully_reclaimable():
    tree = cache()
    tail = insert(tree, list(range(10)))
    tree.tool_call_retention.protect(tail)
    prefix, length, _ = tree.match_prefix(torch.arange(3), update_refs=True)
    assert length == 3
    assert tree.tool_call_retention.protected_tokens == 10
    assert tree.tool_call_retention.reserved_tokens == 7
    tree.dec_node_ref_counter(prefix)
    assert tree.tool_call_retention.reserved_tokens == 10
    tree.evict(10, lambda _: None)
    assert tree.tool_call_retention.reserved_tokens == 0
    assert tree.tool_call_reserved_tokens.arr[0] == 0


def test_router_client_reads_exact_reservation_and_disabled_client_stays_zero(monkeypatch):
    from lightllm.server.router.dynamic_prompt.radix_cache import RadixCacheReadOnlyClient

    name = f"reservation_client_{uuid.uuid4().hex}"
    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: name)
    tree = cache()
    enabled = RadixCacheReadOnlyClient(40, 1, 1, tool_call_retention_enabled=True)
    disabled = RadixCacheReadOnlyClient(40, 1, 1)
    tree.tool_call_retention.protect(insert(tree, [1, 2, 3, 4]))
    assert enabled.get_tool_call_reserved_tokens(0) == 4
    assert disabled.get_tool_call_reserved_tokens(0) == 0
    tree.tool_call_retention.advance_clock(110)
    assert enabled.get_tool_call_reserved_tokens(0) == 0
