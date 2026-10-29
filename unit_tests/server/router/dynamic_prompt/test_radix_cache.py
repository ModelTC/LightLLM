from types import SimpleNamespace

from sortedcontainers import SortedDict

import pytest
import torch

from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache, TreeNode
from lightllm.server.router.dynamic_prompt.hybrid_att_radix_cache import HybridAttPagedRadixCache
from lightllm.utils import shm_utils

pytestmark = pytest.mark.usefixtures("kv_allocator_environment")


@pytest.fixture(scope="module", autouse=True)
def service_name():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(shm_utils, "get_unique_server_name", lambda: "test_radix_cache_service_0")
    yield
    monkeypatch.undo()


def test_case1():
    tree = RadixCache(100, 0)
    ans, _ = tree.insert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"))
    assert ans == 0
    tree.print_self()
    ans, _ = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
    assert ans == 5
    tree.print_self()
    ans, _ = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
    assert ans == 8
    tree.print_self()

    assert tree.get_refed_tokens_num() == 0
    assert tree.get_tree_total_tokens_num() == 13

    # print("evict")
    tree.evict(9, lambda x: x)
    tree.print_self()
    assert tree.get_refed_tokens_num() == 0 and tree.get_tree_total_tokens_num() == 0


def test_case2():
    tree = RadixCache(100, 1)
    ans, _ = tree.insert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"))
    ans, _ = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
    tree.print_self()

    tree_node, size, values = tree.match_prefix(
        torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device="cpu"), update_refs=False
    )
    assert tree_node.node_prefix_total_len == 5 and size == 5 and len(values) == 5
    tree_node, size, values = tree.match_prefix(
        torch.tensor([0, 1, 2, 3, 4, 9], dtype=torch.int64, device="cpu"), update_refs=False
    )
    assert tree_node.node_prefix_total_len == 5 and size == 5 and len(values) == 5
    tree_node, size, values = tree.match_prefix(
        torch.tensor([0, 1, 2, 3, 4, 7, 8], dtype=torch.int64, device="cpu"), update_refs=False
    )
    assert tree_node.node_prefix_total_len == 7 and size == 7 and len(values) == 7
    tree_node, size, values = tree.match_prefix(
        torch.tensor([0, 1, 2, 3, 4, 7, 9], dtype=torch.int64, device="cpu"), update_refs=False
    )
    assert tree_node.node_prefix_total_len == 6 and size == 6 and len(values) == 6
    print(ans)
    return


def test_case3():
    tree = RadixCache(100, 2)
    ans, _ = tree.insert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"))
    ans, _ = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
    tree.print_self()

    tree_node, size, values = tree.match_prefix(
        torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device="cpu"), update_refs=True
    )
    assert tree_node.node_prefix_total_len == 5 and size == 5 and len(values) == 5
    assert tree.get_refed_tokens_num() == 5 and tree.get_tree_total_tokens_num() == 13

    tree_node, size, values = tree.match_prefix(
        torch.tensor([0, 1, 2, 3, 4, 7, 9], dtype=torch.int64, device="cpu"), update_refs=True
    )
    assert tree_node.node_prefix_total_len == 6 and size == 6 and len(values) == 6
    assert tree.get_refed_tokens_num() == 6 and tree.get_tree_total_tokens_num() == 13

    tree.print_self()
    tree.evict(2, lambda x: x)
    assert tree.get_refed_tokens_num() == 6 and tree.get_tree_total_tokens_num() == 8
    tree.print_self()

    tree.dec_node_ref_counter(tree_node)
    tree.print_self()
    print(ans)
    return


def test_case4():
    tree = RadixCache(100, 2)
    ans, _ = tree.insert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"))
    ans, _ = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
    tree.print_self()

    tree.clear_tree_nodes()
    print(ans)
    return


def test_case5():
    """
    测试场景：一个简单的父子节点链 (A -> B)，在 ref_counter 都为 0 时，应该成功合并。
    """
    print("\nTest Case 5: Merging simple parent-child nodes when ref_counter is 0\n")
    tree = RadixCache(100, 0)

    _, node_a = tree.insert(torch.tensor([1, 2, 3], dtype=torch.int64))
    _, node_b = tree.insert(torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64))
    tree.print_self()

    # 验证初始状态：A -> B 结构，且 ref_counter 均为 0
    assert node_b.parent == node_a
    assert torch.equal(node_a.token_id_key, torch.tensor([1, 2, 3], dtype=torch.int64))
    assert len(node_a.children) == 1
    assert node_a.ref_counter == 0
    assert node_b.ref_counter == 0
    assert tree.get_tree_total_tokens_num() == 5

    # 执行合并
    tree.merge_unreferenced_nodes()
    tree.print_self()

    assert torch.equal(node_b.token_id_key, torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64))
    assert node_b.is_leaf()
    assert tree.get_tree_total_tokens_num() == 5
    assert tree.root_node.children[1] is node_b


def test_case6():
    """
    测试场景：一个长的节点链 (A -> B -> C)，在 ref_counter 都为 0 时，应该级联合并成一个节点。
    """
    print("\nTest Case 6: Merging long nodes when ref_counter is 0\n")
    tree = RadixCache(100, 0)
    _, node_a = tree.insert(torch.tensor([1], dtype=torch.int64))
    _, node_b = tree.insert(torch.tensor([1, 2], dtype=torch.int64))
    _, node_c = tree.insert(torch.tensor([1, 2, 3, 4], dtype=torch.int64))
    tree.print_self()

    assert node_c.parent == node_b
    assert node_b.parent == node_a
    assert tree.get_tree_total_tokens_num() == 4
    tree.merge_unreferenced_nodes()
    tree.print_self()

    assert len(tree.root_node.children) == 1
    # 节点 C 的 key 应该是完整的 [1, 2, 3, 4]
    assert torch.equal(node_c.token_id_key, torch.tensor([1, 2, 3, 4], dtype=torch.int64))
    assert node_c.is_leaf()
    assert tree.get_tree_total_tokens_num() == 4


def test_case7():
    """
    测试场景：由于父节点或子节点的 ref_counter > 0，合并不应该发生。
    """
    print("\nTest Case 7: Merging when parent or child ref_counter > 0\n")
    tree = RadixCache(100, 0)

    _, node_a = tree.insert(torch.tensor([1, 2, 3], dtype=torch.int64))
    _, node_b = tree.insert(torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64))
    tree.print_self()

    matched_node, _, _ = tree.match_prefix(torch.tensor([1, 2, 3], dtype=torch.int64), update_refs=True)
    assert matched_node is node_a
    assert node_a.ref_counter == 1
    assert node_b.ref_counter == 0

    tree.merge_unreferenced_nodes()
    tree.print_self()

    assert torch.equal(node_a.token_id_key, torch.tensor([1, 2, 3], dtype=torch.int64))
    assert not node_a.is_leaf()
    assert node_b.parent is node_a


def test_case8():
    """
    测试场景：由于父节点有多个子节点，合并不应该发生。
    """
    print("\nTest Case 8: Merging when parent has multiple children\n")
    tree = RadixCache(100, 0)

    _, node_a = tree.insert(torch.tensor([1, 2], dtype=torch.int64))
    _, node_b = tree.insert(torch.tensor([1, 2, 3], dtype=torch.int64))
    _, node_c = tree.insert(torch.tensor([1, 2, 4], dtype=torch.int64))
    tree.print_self()

    assert len(node_a.children) == 2
    assert node_a.ref_counter == 0
    assert node_b.ref_counter == 0
    assert node_c.ref_counter == 0

    tree.merge_unreferenced_nodes()
    tree.print_self()

    assert len(node_a.children) == 2
    assert torch.equal(node_a.token_id_key, torch.tensor([1, 2], dtype=torch.int64))
    assert tree.root_node.children[1].children[3] is node_b
    assert tree.root_node.children[1].children[4] is node_c


def test_case9():
    """
    测试场景：在一个复杂的树中，只有满足条件的分支被合并。
    """
    print("\nTest Case 9: Merging in a complex tree with mixed conditions\n")
    tree = RadixCache(100, 0)

    # 分支1: 可合并的链 A -> B
    _, node_a = tree.insert(torch.tensor([1, 2], dtype=torch.int64))
    _, node_b = tree.insert(torch.tensor([1, 2, 3], dtype=torch.int64))

    # 分支2: 不可合并的链 C -> D (因为 C 被引用)
    _, node_c = tree.insert(torch.tensor([4, 5], dtype=torch.int64))
    _, node_d = tree.insert(torch.tensor([4, 5, 6], dtype=torch.int64))

    # 增加 C 的引用计数
    tree.match_prefix(torch.tensor([4, 5], dtype=torch.int64), update_refs=True)
    assert node_c.ref_counter == 1
    tree.print_self()

    tree.merge_unreferenced_nodes()
    tree.print_self()

    merged_node_b = tree.root_node.children[1]
    assert torch.equal(merged_node_b.token_id_key, torch.tensor([1, 2, 3], dtype=torch.int64))
    assert merged_node_b.is_leaf()

    unmerged_node_c = tree.root_node.children[4]
    assert torch.equal(unmerged_node_c.token_id_key, torch.tensor([4, 5], dtype=torch.int64))
    assert not unmerged_node_c.is_leaf()
    assert len(unmerged_node_c.children) == 1

    unmerged_node_d = unmerged_node_c.children[6]
    assert torch.equal(unmerged_node_d.token_id_key, torch.tensor([6], dtype=torch.int64))


def test_flush_cache_requires_released_references_and_returns_allocator_pages():
    from lightllm.common.att_resources.allocators import FullAttKVAllocator

    allocator = FullAttKVAllocator(100, pin_memory=False)
    tree = RadixCache(100, 0, att_resource_manager=SimpleNamespace(full_att_allocator=allocator))
    values = allocator.alloc(5)
    key = torch.arange(5)
    tree.insert(key[:3], values[:3])
    tree.insert(key, values)
    node, matched_len, _ = tree.match_prefix(key[:3], update_refs=True)
    assert matched_len == 3
    with pytest.raises(AssertionError, match="can not free"):
        tree.flush_cache()
    assert allocator.can_use_mem_size == 95
    tree.dec_node_ref_counter(node)
    tree.flush_cache()
    assert allocator.can_use_mem_size == 100
    assert tree.get_tree_total_tokens_num() == tree.get_refed_tokens_num() == 0
    assert not tree.root_node.children
    assert tree.root_node.ref_counter == 1


def test_hybrid_flush_releases_shared_kv_and_state_slots():
    from unit_tests.common.att_resources.test_manager_helpers import composition

    resources = composition(with_tail=False)
    allocator = resources.full_att_allocator
    big = resources.bounded_att_big_page_allocator.alloc(1)[0]
    small = resources.bounded_att_small_page_allocator.alloc(1)[0]
    resources.save_bounded_att_pages([0], page_kind="big", slot_ids=[big])
    resources.save_bounded_att_pages([1], page_kind="small", slot_ids=[small])
    values = allocator.alloc(12)
    tree = HybridAttPagedRadixCache(12, 106, 4, 2, resources)
    tree.insert(
        torch.arange(12),
        values,
        block_hashs=[11, 22, 33],
        block_state_idxs=[None, None, small],
        len_to_big_page_id=SortedDict({8: big}),
    )
    node, matched_len, _ = tree.match_prefix(torch.arange(12), block_hashs=[11, 22, 33], update_refs=True)
    assert matched_len == 12
    with pytest.raises(AssertionError, match="can not free"):
        tree.flush_cache()
    tree.dec_node_ref_counter(node)
    tree.flush_cache()
    assert allocator.can_use_mem_size == 12
    assert resources.bounded_att_big_page_allocator.free_count == 3
    assert resources.bounded_att_small_page_allocator.free_count == 2
    assert not tree.root_node.children


def test_page_aligned_insert_and_match():
    tree = RadixCache(100, 99, page_size=4)
    values = torch.arange(100, 110, dtype=torch.int64)

    prefix_len, _ = tree.insert(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), values)
    assert prefix_len == 0
    assert tree.get_tree_total_tokens_num() == 8

    # A mismatch inside a page cannot produce a partial-page cache hit.
    node, matched_len, matched_values = tree.match_prefix(torch.tensor([1, 2, 0, 4, 5, 6, 7, 8]))
    assert node is None
    assert matched_len == 0
    assert matched_values is None

    node, matched_len, matched_values = tree.match_prefix(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 11]))
    assert node is not None
    assert matched_len == 8
    assert matched_values.tolist() == list(range(100, 108))

    # The second sequence shares exactly one page and then branches by its
    # complete second-page key.
    prefix_len, _ = tree.insert(
        torch.tensor([1, 2, 3, 4, 5, 6, 0, 8]),
        torch.arange(200, 208, dtype=torch.int64),
    )
    assert prefix_len == 4
    assert tree.get_tree_total_tokens_num() == 12


def test_page_size_must_match_allocator():
    mem_manager = SimpleNamespace(page_size=8)

    with pytest.raises(ValueError, match="must match allocator page_size 8"):
        RadixCache(100, 100, att_resource_manager=SimpleNamespace(full_att_allocator=mem_manager), page_size=4)


@pytest.mark.parametrize("update_refs", [False, True])
@pytest.mark.parametrize(
    "spans,expected",
    [
        ([], 24),
        ([(10, 21), (22, 30)], 0),
        ([(0, 10), (11, 30)], 0),
        ([(8, 24)], 24),
        ([(24, 30)], 24),
    ],
)
def test_mm_gpu_match_uses_existing_ancestors_and_preserves_other_refs(update_refs, spans, expected):
    tree = RadixCache(64, 103, page_size=4)
    key = torch.arange(32)
    values = key + 100
    tree.insert(key, values)
    held_node, _, _ = tree.match_prefix(key[:24], update_refs=True)

    node, matched_len, matched_values = tree.match_prefix(key[:27], update_refs=update_refs, mm_input_spans=spans)
    assert matched_len == expected
    if expected:
        assert node.node_prefix_total_len == expected
        assert torch.equal(matched_values, values[:expected])
    else:
        assert node is None and matched_values is None
    assert tree.get_refed_tokens_num() == 24

    tree.dec_node_ref_counter(held_node)
    assert tree.get_refed_tokens_num() == (expected if update_refs else 0)
    if update_refs:
        tree.dec_node_ref_counter(node)
    assert tree.get_refed_tokens_num() == 0
    assert tree.root_node.ref_counter == 1
    freed = []
    tree.evict(32, lambda indexes: freed.extend(indexes.tolist()))
    assert sorted(freed) == values.tolist()


@pytest.mark.parametrize("update_refs", [False, True])
@pytest.mark.parametrize("block_start,expected", [(1, 0), (5, 4), (9, 4), (13, 12)])
def test_mm_gpu_fallback_walks_parents_without_revisiting(monkeypatch, update_refs, block_start, expected):
    tree = RadixCache(64, 105, page_size=4)
    tree.insert(torch.arange(40, 44), torch.arange(200, 204))
    key = torch.arange(24)
    values = key + 100
    cached_nodes = {}
    for length in (4, 12, 24):
        _, cached_nodes[length] = tree.insert(key[:length], values[:length])

    # Retreat across existing nodes, even when the image starts inside an ancestor.
    visited = []
    update_time = TreeNode.update_time

    def record_visit(node):
        visited.append(node)
        update_time(node)

    monkeypatch.setattr(TreeNode, "update_time", record_visit)
    node, matched_len, matched_values = tree.match_prefix(
        key, update_refs=update_refs, mm_input_spans=[(block_start, 30)]
    )
    assert matched_len == expected
    assert node is cached_nodes.get(expected)
    assert [len(n.token_id_key) for n in cached_nodes.values()] == [4, 8, 12]
    assert len(visited) == len(set(visited)) == 4
    assert tree.get_refed_tokens_num() == (expected if update_refs else 0)
    if expected:
        assert node.node_prefix_total_len == expected
        assert torch.equal(matched_values, values[:expected])
    else:
        assert node is None and matched_values is None

    # The untouched branch is still oldest; changed tail refs must preserve eviction ordering.
    freed = []
    tree.evict(4, lambda indexes: freed.extend(indexes.tolist()))
    assert freed == list(range(200, 204))
    if update_refs:
        tree.dec_node_ref_counter(node)
    assert tree.get_refed_tokens_num() == 0 and tree.root_node.ref_counter == 1
    tree.evict(24, lambda indexes: freed.extend(indexes.tolist()))
    assert sorted(freed) == sorted(values.tolist() + list(range(200, 204)))
    assert not tree.root_node.children


def test_page_key_bytes_does_not_share_tensor_memory():
    token_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    expected_key = token_ids.clone().numpy().tobytes()
    key = TreeNode(page_size=4).get_child_key(token_ids)

    token_ids[0] = 100

    assert isinstance(key, bytes)
    assert key == expected_key
    assert key != TreeNode(page_size=4).get_child_key(token_ids)


def hybrid_resources(*, freed_big=None, freed_small=None):
    return SimpleNamespace(
        bounded_att_big_page_allocator=SimpleNamespace(
            free=lambda slots: freed_big.extend(slots) if freed_big is not None else None
        ),
        bounded_att_small_page_allocator=SimpleNamespace(
            free_count=0, free=lambda slots: freed_small.extend(slots) if freed_small is not None else None
        ),
    )


def test_hybrid_match_refs_only_valid_prefix_after_tail_state_eviction():
    freed_state_ids = []
    tree = HybridAttPagedRadixCache(
        total_token_num=100,
        rank_in_node=101,
        hash_page_size=4,
        big_page_num=2,
        att_resource_manager=hybrid_resources(freed_small=freed_state_ids),
    )
    key = torch.arange(12, dtype=torch.int64)
    values = torch.arange(100, 112, dtype=torch.int64)
    tree.insert(
        key,
        values,
        block_hashs=[11, 22, 33],
        block_state_idxs=[None, None, 7],
        len_to_big_page_id=SortedDict({8: 5}),
    )

    big_page = tree.root_node.children[22]
    tail_page = big_page.children[33]
    node, matched_len, _ = tree.match_prefix(key, block_hashs=[11, 22, 33], update_refs=True)
    assert node is tail_page and matched_len == 12
    assert tree.get_refed_tokens_num() == 12
    tree.dec_node_ref_counter(node)
    assert tree.get_refed_tokens_num() == 0

    tree.free_one_small_page_buffer()
    assert freed_state_ids == [7]
    node, matched_len, matched_values = tree.match_prefix(key, block_hashs=[11, 22, 33], update_refs=True)
    assert node is big_page and matched_len == 8
    assert matched_values.tolist() == list(range(100, 108))
    assert big_page.ref_counter == 1 and tail_page.ref_counter == 0
    assert tree.get_refed_tokens_num() == 8
    tree.dec_node_ref_counter(node)
    assert big_page.ref_counter == tail_page.ref_counter == tree.get_refed_tokens_num() == 0
    assert tree.root_node.ref_counter == 1


@pytest.fixture
def hybrid_tree():
    freed_big_states = []
    tree = HybridAttPagedRadixCache(
        total_token_num=100,
        rank_in_node=102,
        hash_page_size=4,
        big_page_num=2,
        att_resource_manager=hybrid_resources(freed_big=freed_big_states),
    )
    return tree, freed_big_states


def test_hybrid_match_traverses_pages_inside_image(hybrid_tree):
    tree, freed = hybrid_tree
    key = torch.arange(20)
    tree.insert(
        key,
        key + 100,
        block_hashs=[1, 2, 3, 4, 5],
        block_state_idxs=[None, None, None, None, 7],
        len_to_big_page_id=SortedDict({8: 5, 16: 6}),
    )
    first_page = tree.root_node.children[2]
    second_page = first_page.children[4]
    assert tree.get_big_page_ids_by_node(second_page) == [5, 6]

    for length, expected in [(8, 0), (12, 0), (16, 16), (20, 20), (24, 20)]:
        node, matched, values = tree.match_prefix(
            torch.arange(length),
            block_hashs=list(range(1, length // 4 + 1)),
            update_refs=True,
            mm_input_spans=[(4, 12)],
        )
        assert matched == expected
        if expected:
            assert not 4 < node.node_prefix_total_len < 12
            assert values.tolist() == list(range(100, 100 + expected))
        assert tree.get_refed_tokens_num() == expected
        tree.dec_node_ref_counter(node)
        assert tree.get_refed_tokens_num() == 0

    # Duplicate requests release their slots; no per-page validity metadata is needed.
    tree.insert(key[:8], key[:8] + 100, [1, 2], [None, None], SortedDict({8: 8}))
    tree.insert(key[:16], key[:16] + 100, [1, 2, 3, 4], [None] * 4, SortedDict({8: 9, 16: 10}))
    assert first_page.big_page_buffer_idx == 5
    assert second_page.big_page_buffer_idx == 6
    assert freed == [8, 9, 10]
    tree._evict(20, lambda *args: None)
    assert sorted(freed) == [5, 6, 8, 9, 10]
    assert not tree.root_node.children


@pytest.mark.parametrize(
    "small_state_valid,mm_spans,match_len,expected",
    [
        (True, (), 640, 512),
        (True, [(448, 640)], 384, 384),
        (True, [(448, 640)], 640, 256),
        (False, [(448, 640)], 640, 256),
        (True, [(448, 640)], 768, 768),
        (True, [(448, 896)], 896, 256),
        (False, [(448, 896)], 896, 256),
        (True, [(448, 896)], 1024, 1024),
        (True, [(700, 800)], 896, 512),
    ],
)
def test_hybrid_match_falls_back_along_matched_branch(monkeypatch, small_state_valid, mm_spans, match_len, expected):
    freed_big_states = []
    tree = HybridAttPagedRadixCache(
        total_token_num=2048,
        rank_in_node=104,
        hash_page_size=64,
        big_page_num=4,
        att_resource_manager=hybrid_resources(freed_big=freed_big_states),
    )
    # The older request leaves a small-page state at 384, before the image.
    old_values = torch.arange(1000, 1384)
    tree.insert(
        torch.arange(384),
        old_values,
        block_hashs=list(range(1, 7)),
        block_state_idxs=[None] * 5 + [7],
        len_to_big_page_id=SortedDict({256: 5}),
    )
    # A longer request keeps the old small branch and owns a separate big-page tail.
    new_values = torch.cat([old_values[:256], torch.arange(2000, 2768)])
    big_slots = {512: 6, 768: 8, 1024: 9}
    prefix_len, _ = tree.insert(
        torch.arange(1024),
        new_values,
        block_hashs=list(range(1, 17)),
        block_state_idxs=[None] * 16,
        len_to_big_page_id=SortedDict(big_slots),
    )
    assert prefix_len == 256
    if not small_state_valid:
        tree.free_one_small_page_buffer()

    common = tree.root_node.children[4]
    small_320 = common.children[5]
    small_384 = small_320.children[6]
    big_512 = common.children[8]
    big_768 = big_512.children[12]
    big_1024 = big_768.children[16]
    visited = []
    update_time = type(common).update_time

    def record_visit(node):
        visited.append(node)
        update_time(node)

    monkeypatch.setattr(type(common), "update_time", record_visit)
    node, matched_len, values = tree.match_prefix(
        torch.arange(match_len),
        block_hashs=list(range(1, match_len // 64 + 1)),
        update_refs=True,
        mm_input_spans=mm_spans,
    )
    assert matched_len == expected
    # Retain small pages only when already on the matched path; never rematch sibling branches.
    assert len(visited) == len(set(visited))
    assert not any(start < matched_len < end for start, end in mm_spans)
    expected_values = old_values if expected <= 384 else new_values
    assert torch.equal(values, expected_values[:expected])
    assert tree.get_refed_tokens_num() == expected
    assert tree.root_node.ref_counter == 2 and common.ref_counter == 1
    assert small_320.ref_counter == small_384.ref_counter == int(expected == 384)
    for big in (big_512, big_768, big_1024):
        assert big.ref_counter == int(big.node_prefix_total_len <= expected)

    tree.dec_node_ref_counter(node)
    assert tree.get_refed_tokens_num() == 0 and tree.root_node.ref_counter == 1
    assert all(n.ref_counter == 0 for n in (common, small_320, small_384, big_512, big_768, big_1024))
    freed_indexes = []
    tree._evict(tree.get_tree_total_tokens_num(), lambda indexes, _: freed_indexes.extend(indexes.tolist()))
    assert sorted(freed_indexes) == sorted(old_values.tolist() + new_values[256:].tolist())
    assert sorted(freed_big_states) == [5, 6, 8, 9]
    assert not tree.root_node.children


@pytest.mark.parametrize("first_outside_image", [False, True])
@pytest.mark.parametrize("require_state", [False, True])
def test_hybrid_fallback_distinguishes_kv_ownership_from_state_restore(hybrid_tree, first_outside_image, require_state):
    tree, freed = hybrid_tree
    tree.insert(
        torch.arange(20),
        block_hashs=[1, 2, 3, 4, 5],
        block_state_idxs=[None] * 4 + [7],
        len_to_big_page_id=SortedDict({8: 6, 16: 8}),
    )
    spans = [(12, 18)] if first_outside_image else [(4, 18)]
    node, _, _ = tree.match_prefix(
        torch.arange(20), block_hashs=[1, 2, 3, 4, 5], update_refs=True, mm_input_spans=spans
    )
    node = tree.deref_to_first_big_page_node(node, mm_input_spans=spans if require_state else ())
    expected = 16 if not require_state else (8 if first_outside_image else 0)
    assert (0 if node is None else node.node_prefix_total_len) == expected
    assert tree.get_refed_tokens_num() == expected
    tree.dec_node_ref_counter(node)
    assert tree.get_refed_tokens_num() == 0
    tree._evict(20, lambda *args: None)
    assert sorted(freed) == [6, 8]  # Invalid state slots have the same ownership and lifetime.


@pytest.mark.parametrize(
    "computed,cache_len,retained,expected",
    [(12, 20, 0, 0), (18, 20, 16, 16), (20, 20, 20, 20), (26, 20, 20, 20), (26, 24, 24, 16)],
)
def test_hybrid_request_release_frees_or_transfers_image_page_slots(
    monkeypatch, hybrid_tree, computed, cache_len, retained, expected
):
    from lightllm.server.router.model_infer import infer_batch
    from lightllm.server.router.model_infer.infer_batch import InferReq, InferenceContext

    tree, freed = hybrid_tree
    context = InferenceContext()
    context.args = SimpleNamespace(linear_att_hash_page_size=4, linear_att_page_block_num=2)
    monkeypatch.setattr(infer_batch.g_infer_context, "is_hybrid_att_model", True)
    monkeypatch.setattr(infer_batch, "get_env_start_args", lambda: context.args)
    context.radix_cache = tree
    context.req_manager = SimpleNamespace(req_to_token_indexs=torch.arange(100, 132).view(1, -1))
    req = InferReq.__new__(InferReq)
    req.req_idx = req.cur_output_len = 0
    req.cur_kv_len, req.hold_kv_len, req.hybrid_cache_len = computed, max(24, computed), cache_len
    req.shared_kv_node = None
    req.hybrid_len_to_big_page_id = SortedDict({end: end // 8 + 4 for end in range(8, min(computed, cache_len) + 1, 8)})
    big_state_ids = list(req.hybrid_len_to_big_page_id.values())
    req.mm_input_spans = [(4, 12)] + ([(20, cache_len + 1)] if cache_len >= 24 else [])
    req.tail_small_page_buffer_id = 7 if computed >= cache_len and cache_len % 8 else None
    block_hashs = list(range(1, cache_len // 4 + 1))
    req.shm_req = SimpleNamespace(
        input_len=cache_len + 1,
        shm_prompt_ids=SimpleNamespace(arr=list(range(cache_len + 1))),
        hybrid_token_hash_list=SimpleNamespace(get_all=lambda: block_hashs),
    )
    released = []
    context._hybrid_att_free_req(released, req)
    assert torch.cat(released).tolist() == list(range(100 + retained, 100 + max(24, computed)))
    assert not req.hybrid_len_to_big_page_id
    assert req.tail_small_page_buffer_id is None
    assert tree.get_tree_total_tokens_num() == retained
    assert freed == ([] if retained else big_state_ids)
    node, length, _ = tree.match_prefix(
        torch.arange(cache_len), block_hashs=block_hashs, update_refs=True, mm_input_spans=req.mm_input_spans
    )
    assert length == expected
    assert not any(start < length < end for start, end in req.mm_input_spans)
    tree.dec_node_ref_counter(node)
    tree._evict(retained, lambda *args: None)
    assert sorted(freed) == big_state_ids


if __name__ == "__main__":
    pytest.main()
