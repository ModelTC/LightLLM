import random
import pytest
from lightllm.server.httpserver_for_pd_master.pd_selector import create_selector, CacheAwareSelector
from lightllm.server.pd_io_struct import PD_Client_Obj


def _make_nodes(mode, usages):
    return [
        PD_Client_Obj(node_id=i, client_ip_port=f"127.0.0.1:{8000 + i}", mode=mode, start_args=None)
        for i in range(len(usages))
    ]


def _make_selector(p_usages=(0.0, 0.0), d_usages=(0.0,)):
    selector: CacheAwareSelector = create_selector("cache_aware", None)
    p_nodes = _make_nodes("prefill", p_usages)
    d_nodes = _make_nodes("decode", d_usages)
    for node, usage in zip(p_nodes, p_usages):
        node.run_status.total_token_usage_rate = usage
    for node, usage in zip(d_nodes, d_usages):
        node.run_status.total_token_usage_rate = usage
    selector.update_nodes(p_nodes, d_nodes)
    return selector, p_nodes


def test_multi_turn_affinity():
    random.seed(0)
    selector, p_nodes = _make_selector()
    turn1 = "conv-a-" * 60
    p1, _ = selector.select_p_d_node(turn1, None, None)
    p_nodes[p1.node_id].run_status.total_token_usage_rate = 0.2
    turn2 = turn1 + "assistant reply and next user question " * 10
    for _ in range(20):
        p2, _ = selector.select_p_d_node(turn2, None, None)
        assert p2.node_id == p1.node_id


def test_token_ids_prompt():
    random.seed(0)
    selector, _ = _make_selector()
    turn1 = list(range(300))
    p1, _ = selector.select_p_d_node(turn1, None, None)
    p2, _ = selector.select_p_d_node(turn1 + list(range(100)), None, None)
    assert p2.node_id == p1.node_id


def test_distinct_conversations_spread():
    random.seed(0)
    selector, p_nodes = _make_selector()
    counts = {0: 0, 1: 0}
    for i in range(100):
        prompt = f"conv-{i}-" + "x" * 500
        p, _ = selector.select_p_d_node(prompt, None, None)
        counts[p.node_id] += 1
    assert counts[0] > 20 and counts[1] > 20


def test_overloaded_node_falls_back():
    random.seed(0)
    selector, p_nodes = _make_selector()
    turn1 = "conv-b-" * 60
    p1, _ = selector.select_p_d_node(turn1, None, None)
    other = p_nodes[1 - p1.node_id]
    p_nodes[p1.node_id].run_status.total_token_usage_rate = 0.95
    turn2 = turn1 + "more " * 20
    counts = {p1.node_id: 0, other.node_id: 0}
    for _ in range(50):
        p2, _ = selector.select_p_d_node(turn2, None, None)
        counts[p2.node_id] += 1
        selector.prefix_to_node.clear()
        selector._record(selector._chain_hashes(turn1), p1.node_id)
    assert counts[other.node_id] > counts[p1.node_id]


def test_short_prompt_no_crash():
    random.seed(0)
    selector, p_nodes = _make_selector()
    p, d = selector.select_p_d_node("hi", None, None)
    assert p in p_nodes


def test_lru_capacity_bound():
    selector, _ = _make_selector()
    selector.MAX_ENTRIES = 10
    for i in range(50):
        selector.select_p_d_node(f"conv-{i}-" + "y" * 500, None, None)
    assert len(selector.prefix_to_node) <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
