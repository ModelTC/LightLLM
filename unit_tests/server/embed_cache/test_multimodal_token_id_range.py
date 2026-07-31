from types import SimpleNamespace

from lightllm.server.embed_cache.impl import naive_memory_cache


def test_multimodal_token_id_range_selection_uses_time_seed(monkeypatch):
    seed_values = []
    selected_index = 1234

    monkeypatch.setattr(naive_memory_cache.time, "time", lambda: 123.5)
    monkeypatch.setattr(naive_memory_cache.random, "seed", seed_values.append)
    monkeypatch.setattr(naive_memory_cache.random, "randint", lambda start, end: selected_index)

    cache = naive_memory_cache.InMemoryCache.__new__(naive_memory_cache.InMemoryCache)
    range_index = cache._set_prefill_node_random_token_id_range()
    range_size = ((2 ** 63 - 1) - 100000000) // 10000

    assert seed_values == [123.5]
    assert range_index == selected_index
    assert cache.token_id_range_start == 100000000 + selected_index * range_size
    assert cache.token_id_range_end == cache.token_id_range_start + range_size


def test_cache_manager_selects_and_logs_probabilistic_range(monkeypatch):
    node_uuid = 0x12345678123456781234567812345678
    expected_index = 4321
    range_size = ((2 ** 63 - 1) - 100000000) // 10000
    expected_start = 100000000 + expected_index * range_size
    expected_end = expected_start + range_size
    warning_messages = []

    def set_expected_range(cache):
        cache.token_id_range_start = expected_start
        cache.token_id_range_end = expected_end
        return expected_index

    monkeypatch.setattr(naive_memory_cache, "CpuEmbedCacheClient", lambda **kwargs: object())
    monkeypatch.setattr(
        naive_memory_cache.InMemoryCache,
        "_set_prefill_node_random_token_id_range",
        set_expected_range,
    )
    monkeypatch.setattr(naive_memory_cache.logger, "warning", warning_messages.append)

    cache = naive_memory_cache.InMemoryCache(
        SimpleNamespace(
            cache_capacity=200,
            config_server_host=None,
            config_server_port=None,
            pd_node_id=node_uuid,
            run_mode="prefill",
        )
    )
    cache._check_and_set_new_id_range(1)

    assert cache.token_id_range_start == expected_start
    assert cache.token_id_range_end == expected_end
    assert len(warning_messages) == 1
    assert f"slot={expected_index}/9999" in warning_messages[0]
    assert "does not guarantee global uniqueness" in warning_messages[0]
    assert "0.0100% probability" in warning_messages[0]
