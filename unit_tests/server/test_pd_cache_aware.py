import pytest

from lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree import PromptCacheTree


@pytest.mark.parametrize("operation", ["insert", "prefix_match"])
def test_prompt_cache_tree_rejects_mismatched_sample_stride(operation):
    tree = PromptCacheTree(sample_stride=4)

    with pytest.raises(AssertionError, match="sample_stride mismatch"):
        if operation == "insert":
            tree.insert_sampled("po", "prefill", 2)
        else:
            tree.prefix_match_sampled("po", len("prompt"), 2)


def test_prompt_cache_tree_handles_key_deeper_than_python_recursion_limit():
    tree = PromptCacheTree(sample_stride=1)
    key = "x" * 5000

    tree.insert_sampled(key, "prefill", sample_stride=1)
    result = tree.prefix_match_sampled(key, input_char_count=len(key), sample_stride=1)

    assert result.prefill_node == "prefill"
    assert result.matched_char_count == len(key)


def test_prompt_cache_tree_empty_key_does_not_create_nodes():
    tree = PromptCacheTree(sample_stride=1)

    tree.insert_sampled("", "prefill", sample_stride=1)
    result = tree.prefix_match_sampled("", input_char_count=0, sample_stride=1)

    assert tree._node_count == 0
    assert not tree.root.children
    assert result.prefill_node is None
    assert result.matched_char_count == 0
    assert result.input_char_count == 0
    assert result.last_insert_time is None


def test_prompt_cache_tree_zero_capacity_evicts_the_complete_path():
    tree = PromptCacheTree(sample_stride=1, max_node_count=0, evict_node_batch=1)

    tree.insert_sampled("你好🙂", "prefill", sample_stride=1)
    result = tree.prefix_match_sampled("你好🙂", input_char_count=3, sample_stride=1)

    assert tree._node_count == 0
    assert not tree.root.children
    assert result.prefill_node is None
    assert result.matched_char_count == 0
    assert result.input_char_count == 3
    assert result.last_insert_time is None


def test_prompt_cache_tree_matches_unicode_partial_prefix():
    tree = PromptCacheTree(sample_stride=1)
    tree.insert_sampled("你好🙂", "prefill", sample_stride=1)

    result = tree.prefix_match_sampled("你好🙃", input_char_count=3, sample_stride=1)

    assert result.prefill_node == "prefill"
    assert result.matched_char_count == 2
    assert result.input_char_count == 3
    assert result.last_insert_time is not None


def test_prompt_cache_tree_iterative_insert_keeps_lru_eviction_order():
    tree = PromptCacheTree(sample_stride=1, max_node_count=3, evict_node_batch=1)
    tree.insert_sampled("ab", "prefill-ab", sample_stride=1)
    tree.insert_sampled("ac", "prefill-ac", sample_stride=1)

    # 再次访问 ab，使 ac 成为最早未使用的叶节点。
    tree.insert_sampled("ab", "prefill-ab", sample_stride=1)
    # 插入 d 后超出容量，依次驱逐 ac 和 ab；共同父节点 a 随后成为叶节点。
    tree.insert_sampled("d", "prefill-d", sample_stride=1)

    assert tree._node_count == 2
    assert set(tree.root.children) == {"a", "d"}
    result = tree.prefix_match_sampled("ab", input_char_count=2, sample_stride=1)
    assert result.prefill_node == "prefill-ab"
    assert result.matched_char_count == 1
