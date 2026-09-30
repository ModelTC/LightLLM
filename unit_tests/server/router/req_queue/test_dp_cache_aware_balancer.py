from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from lightllm.server.api_cli import make_argument_parser
from lightllm.server.router.req_queue import dp_balancer
from lightllm.server.router.req_queue.dp_balancer.cache_aware import (
    DpCacheAwareBalancer,
    DpCacheAwareConfig,
    TokenPrefixCache,
)


class Queue:
    def __init__(self):
        self.waiting_req_list = []

    def extend(self, reqs):
        self.waiting_req_list.extend(reqs)


def req(tokens, disabled=False):
    tokens = np.array(tokens, dtype=np.int64)
    return SimpleNamespace(
        input_len=len(tokens),
        sample_params=SimpleNamespace(disable_prompt_cache=disabled),
        shm_prompt_ids=object(),
        get_prompt_ids_numpy=lambda: tokens,
    )


@pytest.mark.parametrize("hybrid, expected", [(False, 16), (True, 512)])
def test_factory_uses_existing_model_cache_page_size(monkeypatch, hybrid, expected):
    monkeypatch.setattr(dp_balancer, "is_hybrid_att_model", lambda _path: hybrid)
    args = SimpleNamespace(
        dp_balancer="cache_aware",
        disable_dynamic_prompt_cache=False,
        model_dir="model",
        page_size=16,
        linear_att_hash_page_size=512,
    )
    balancer = dp_balancer.get_dp_balancer(args, 2, [Queue(), Queue()])
    assert balancer.config.block_size == expected
    args.disable_dynamic_prompt_cache = True
    with pytest.raises(ValueError, match="requires dynamic prompt cache"):
        dp_balancer.get_dp_balancer(args, 2, [Queue(), Queue()])


def test_cli_preserves_default_and_accepts_opt_in_strategy():
    parser = make_argument_parser()
    assert parser.parse_args([]).dp_balancer == "bs_balancer"
    assert parser.parse_args(["--dp_balancer", "cache_aware"]).dp_balancer == "cache_aware"


def test_prefix_hashes_exclude_last_token_and_match_longest_aligned_prefix():
    cache = TokenPrefixCache(4, 10, 1)
    assert not cache.hash_prefixes(np.arange(4, dtype=np.int64))
    first = cache.hash_prefixes(np.arange(9, dtype=np.int64))
    cache.insert(first, 1)
    assert cache.match(first) == (1, 8)
    changed = np.arange(9, dtype=np.int64)
    changed[5] = 100
    assert cache.match(cache.hash_prefixes(changed)) == (1, 4)


def test_prefix_index_is_bounded_and_evicts_least_recently_used():
    cache = TokenPrefixCache(1, 2, 1)
    cache.insert([(1, 1), (2, 2)], 0)
    assert cache.match([(1, 1)]) == (0, 1)
    cache.insert([(3, 1)], 1)
    assert len(cache) <= 2
    assert cache.match([(2, 2)]) == (None, 0)
    assert cache.match([(3, 1)]) == (1, 1)


def test_affinity_preserves_request_group_and_yields_to_overload():
    queues = [Queue(), Queue()]
    balancer = DpCacheAwareBalancer(2, queues, DpCacheAwareConfig(block_size=4))
    tokens = list(range(9))
    balancer.prefix_cache.insert(balancer.prefix_cache.hash_prefixes(np.array(tokens, dtype=np.int64)), 1)
    group = [req(tokens), req(tokens)]
    pending = [group]
    balancer.assign_reqs_to_dp(SimpleNamespace(get_all_dp_req_num=lambda: [1, 1]), pending)
    assert not pending and queues[1].waiting_req_list == group
    assert all(item.sample_params.suggested_dp_index == 1 for item in group)
    overloaded = req(tokens)
    balancer.assign_reqs_to_dp(SimpleNamespace(get_all_dp_req_num=lambda: [1, 20]), [[overloaded]])
    assert overloaded.sample_params.suggested_dp_index == 0


def test_disabled_cache_does_not_attach_prompt_or_publish_affinity():
    queues = [Queue(), Queue()]
    balancer = DpCacheAwareBalancer(2, queues, DpCacheAwareConfig(block_size=4))
    request = req(range(9), disabled=True)
    del request.shm_prompt_ids
    request.link_prompt_ids_shm_array = Mock(side_effect=AssertionError("must not attach"))
    balancer.assign_reqs_to_dp(None, [[request]])
    assert len(balancer.prefix_cache) == 0


def test_router_releases_only_prompt_mapping_it_attached():
    balancer = DpCacheAwareBalancer(1, [Queue()], DpCacheAwareConfig(block_size=4))
    request = req(range(9))
    del request.shm_prompt_ids
    mapping = SimpleNamespace(detach_shm=Mock())
    request.link_prompt_ids_shm_array = lambda: setattr(request, "shm_prompt_ids", mapping)
    balancer.assign_reqs_to_dp(None, [[request]])
    mapping.detach_shm.assert_called_once_with()
    assert not hasattr(request, "shm_prompt_ids")
