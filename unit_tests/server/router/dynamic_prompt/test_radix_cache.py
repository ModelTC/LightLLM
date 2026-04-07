import pytest
import torch
from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache


def test_case1():
    tree = RadixCache("unique_name", 100, 0)
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
    tree = RadixCache("unique_name", 100, 1)
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
    tree = RadixCache("unique_name", 100, 2)
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

    tree = RadixCache("unique_name", 100, 2)
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
    tree = RadixCache("unique_name", 100, 0)

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
    tree = RadixCache("unique_name", 100, 0)
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
    tree = RadixCache("unique_name", 100, 0)

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
    tree = RadixCache("unique_name", 100, 0)

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
    tree = RadixCache("unique_name", 100, 0)

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


if __name__ == "__main__":
    pytest.main()
