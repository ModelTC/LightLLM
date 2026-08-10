from types import SimpleNamespace

from lightllm.server.httpserver_for_pd_master.pd_selector.cache_aware import CacheAwarePolicy


def _worker(address: str, dispatched_prompt_chars: int = 0):
    return SimpleNamespace(
        client_ip_port=address,
        dispatched_prompt_chars=dispatched_prompt_chars,
    )


def test_cache_aware_keeps_cache_worker_when_inflight_load_is_balanced():
    policy = CacheAwarePolicy()
    cache_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=110)
    least_loaded_worker = _worker("10.0.0.2:8000", dispatched_prompt_chars=100)
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_worker.client_ip_port)

    selected_worker = policy.select_worker([cache_worker, least_loaded_worker], prompt)

    assert selected_worker is cache_worker


def test_cache_aware_uses_least_loaded_worker_when_cache_worker_is_overloaded():
    policy = CacheAwarePolicy()
    cache_worker = _worker("10.0.0.1:8000", dispatched_prompt_chars=2000)
    least_loaded_worker = _worker("10.0.0.2:8000", dispatched_prompt_chars=100)
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_worker.client_ip_port)

    selected_worker = policy.select_worker([cache_worker, least_loaded_worker], prompt)

    assert selected_worker is least_loaded_worker


def test_cache_aware_keeps_cache_worker_when_both_workers_are_idle():
    policy = CacheAwarePolicy()
    cache_worker = _worker("10.0.0.1:8000")
    other_worker = _worker("10.0.0.2:8000")
    prompt = "shared prefix " * 100
    policy.prompt_cache_tree.insert(prompt, cache_worker.client_ip_port)

    selected_worker = policy.select_worker([other_worker, cache_worker], prompt)

    assert selected_worker is cache_worker
