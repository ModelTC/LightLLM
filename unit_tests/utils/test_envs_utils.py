from lightllm.utils.envs_utils import (
    get_pd_cache_high_priority_max_age_seconds,
    get_pd_node_router_wait_timeout_seconds,
    get_pd_node_shm_req_alloc_timeout_seconds,
)


def test_pd_cache_high_priority_max_age_defaults_to_16_seconds(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_PD_CACHE_HIGH_PRIORITY_MAX_AGE_SECONDS", raising=False)
    get_pd_cache_high_priority_max_age_seconds.cache_clear()

    assert get_pd_cache_high_priority_max_age_seconds() == 16

    get_pd_cache_high_priority_max_age_seconds.cache_clear()


def test_pd_cache_high_priority_max_age_reads_environment_variable(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_PD_CACHE_HIGH_PRIORITY_MAX_AGE_SECONDS", "30")
    get_pd_cache_high_priority_max_age_seconds.cache_clear()

    assert get_pd_cache_high_priority_max_age_seconds() == 30

    get_pd_cache_high_priority_max_age_seconds.cache_clear()


def test_pd_node_shm_req_alloc_timeout_defaults_to_20_seconds(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_PD_NODE_SHM_REQ_ALLOC_TIMEOUT_SECONDS", raising=False)
    get_pd_node_shm_req_alloc_timeout_seconds.cache_clear()

    assert get_pd_node_shm_req_alloc_timeout_seconds() == 20

    get_pd_node_shm_req_alloc_timeout_seconds.cache_clear()


def test_pd_node_shm_req_alloc_timeout_reads_environment_variable(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_PD_NODE_SHM_REQ_ALLOC_TIMEOUT_SECONDS", "30")
    get_pd_node_shm_req_alloc_timeout_seconds.cache_clear()

    assert get_pd_node_shm_req_alloc_timeout_seconds() == 30

    get_pd_node_shm_req_alloc_timeout_seconds.cache_clear()


def test_pd_node_router_wait_timeout_defaults_to_20_seconds(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_PD_NODE_ROUTER_WAIT_TIMEOUT_SECONDS", raising=False)
    get_pd_node_router_wait_timeout_seconds.cache_clear()

    assert get_pd_node_router_wait_timeout_seconds() == 20

    get_pd_node_router_wait_timeout_seconds.cache_clear()


def test_pd_node_router_wait_timeout_reads_environment_variable(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_PD_NODE_ROUTER_WAIT_TIMEOUT_SECONDS", "45")
    get_pd_node_router_wait_timeout_seconds.cache_clear()

    assert get_pd_node_router_wait_timeout_seconds() == 45

    get_pd_node_router_wait_timeout_seconds.cache_clear()
