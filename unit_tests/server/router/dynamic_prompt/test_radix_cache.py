import pytest
import torch
from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache


def test_case1():
    tree = RadixCache("unique_name", 100, 0)
    ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"))
    assert ans == 0
    tree.print_self()
    ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
    assert ans == 5
    tree.print_self()
    ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
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
    ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"))
    ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
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
    ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"))
    ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
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
    ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64, device="cpu"))
    ans = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64, device="cpu"))
    tree.print_self()

    tree.clear_tree_nodes()
    print(ans)
    return


if __name__ == "__main__":
    pytest.main()
