import json

import pytest

from lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_key import prompt_cache_key
from lightllm.server.httpserver_for_pd_master.pd_selector.prompt_cache_tree import PromptCacheTree


@pytest.mark.parametrize("difference", [1, 17, 511, 1025, 2047])
def test_a_difference_between_legacy_sample_points_invalidates_the_first_block(difference):
    first = "a" * 65536
    other = first[:difference] + "b" + first[difference + 1 :]
    assert first[::512] == other[::512]
    tree = PromptCacheTree(sample_stride=2048)
    tree.insert_sampled(prompt_cache_key(first, 2048), "p0/dp3", 2048)
    result = tree.prefix_match_sampled(prompt_cache_key(other, 2048), len(other), 2048)
    assert result.matched_char_count == 0
    assert result.prefill_node is None


@pytest.mark.parametrize("size", [1, 37, 2047, 2048, 2049, 4096, 65535, 1000000])
def test_exact_and_append_only_prefix_counts_are_conservative(size):
    original = ("你好🙂abc" * (size // 6 + 1))[:size]
    tree = PromptCacheTree(sample_stride=2048)
    tree.insert_sampled(prompt_cache_key(original, 2048), "p0/dp7", 2048)
    exact = tree.prefix_match_sampled(prompt_cache_key(original, 2048), size, 2048)
    assert exact.matched_char_count == size
    grown = original + " next turn"
    result = tree.prefix_match_sampled(prompt_cache_key(grown, 2048), len(grown), 2048)
    assert max(0, size - 2047) <= result.matched_char_count <= size


def test_a_changed_later_block_keeps_only_the_verified_contiguous_prefix():
    first = "x" * 10000
    other = first[:5000] + "y" + first[5001:]
    tree = PromptCacheTree(sample_stride=2048)
    tree.insert_sampled(prompt_cache_key(first, 2048), "p1/dp2", 2048)
    result = tree.prefix_match_sampled(prompt_cache_key(other, 2048), len(other), 2048)
    assert result.matched_char_count == 4096


def test_million_token_english_hint_stays_below_the_metadata_frame_budget():
    # Seven million characters conservatively covers the measured English input.
    key = prompt_cache_key("abcdefg" * 1000000, 2048)
    assert len(json.dumps({"key": key}).encode()) < 96 * 1024
    assert len(key) == 3418


def test_empty_and_invalid_block_size():
    assert prompt_cache_key("", 2048) == []
    with pytest.raises(ValueError):
        prompt_cache_key("x", 0)
