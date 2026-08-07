"""cache-aware PD prefill 选点器均衡门闩的单测：验证近期派发量（半衰期衰减）能打破 cache 亲和导致的单点饥饿。"""
import time

import pytest

from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.server.httpserver_for_pd_master.pd_selector.cache_aware import (
    CacheAwareConfig,
    CacheAwarePolicy,
)


def _worker(name: str) -> PD_Client_Obj:
    return PD_Client_Obj(node_id=0, client_ip_port=name, mode="prefill", start_args={})


PROMPT = "Explain paged KV cache. " * 200  # long enough to span a sample stride


def test_decay_math():
    cfg = CacheAwareConfig(balance_half_life_secs=60.0)
    policy = CacheAwarePolicy(cfg)
    w = _worker("A:1")
    w.recent_dispatched_chars = 1000.0
    w.last_decay_ts = time.monotonic() - 120.0  # two half-lives idle
    policy._decay_recent([w])
    assert w.recent_dispatched_chars == pytest.approx(250.0, rel=1e-3)


def test_starvation_redirects_to_idle_node():
    cfg = CacheAwareConfig(balance_rel_threshold=1.2, balance_half_life_secs=60.0)
    policy = CacheAwarePolicy(cfg)
    a, b = _worker("A:1"), _worker("B:1")
    policy.prompt_cache_tree.insert(PROMPT, a.client_ip_port)
    a.recent_dispatched_chars = 1000.0
    b.recent_dispatched_chars = 1000.0
    now = time.monotonic()
    a.last_decay_ts = now
    b.last_decay_ts = now - 120.0
    chosen = policy.select_worker([a, b], request_text=PROMPT)
    assert chosen.client_ip_port == "B:1"


def test_balanced_keeps_cache_affinity():
    cfg = CacheAwareConfig(balance_rel_threshold=1.2, balance_half_life_secs=60.0)
    policy = CacheAwarePolicy(cfg)
    a, b = _worker("A:1"), _worker("B:1")
    policy.prompt_cache_tree.insert(PROMPT, a.client_ip_port)
    a.recent_dispatched_chars = 1000.0
    b.recent_dispatched_chars = 1000.0
    now = time.monotonic()
    a.last_decay_ts = now
    b.last_decay_ts = now
    chosen = policy.select_worker([a, b], request_text=PROMPT)
    assert chosen.client_ip_port == "A:1"
