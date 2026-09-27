from lightllm.utils.envs_utils import (
    get_eplb_transfer_layer_parallelism,
    get_pd_cache_high_priority_max_age_seconds,
    get_pd_cache_high_priority_min_prompt_tokens,
    get_pd_node_busy_retry_timeout_seconds,
    get_pd_node_continuation_resource_wait_timeout_seconds,
    get_pd_node_resource_wait_timeout_seconds,
)


def test_eplb_transfer_layer_parallelism_defaults_to_sixteen(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_EPLB_TRANSFER_LAYER_PARALLELISM", raising=False)
    get_eplb_transfer_layer_parallelism.cache_clear()

    assert get_eplb_transfer_layer_parallelism() == 16

    get_eplb_transfer_layer_parallelism.cache_clear()


def test_eplb_transfer_layer_parallelism_reads_environment_variable(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_EPLB_TRANSFER_LAYER_PARALLELISM", "4")
    get_eplb_transfer_layer_parallelism.cache_clear()

    assert get_eplb_transfer_layer_parallelism() == 4

    get_eplb_transfer_layer_parallelism.cache_clear()


def test_pd_cache_high_priority_max_age_defaults_to_36_seconds(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_PD_CACHE_HIGH_PRIORITY_MAX_AGE_SECONDS", raising=False)
    get_pd_cache_high_priority_max_age_seconds.cache_clear()

    assert get_pd_cache_high_priority_max_age_seconds() == 36

    get_pd_cache_high_priority_max_age_seconds.cache_clear()


def test_pd_cache_high_priority_max_age_reads_environment_variable(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_PD_CACHE_HIGH_PRIORITY_MAX_AGE_SECONDS", "30")
    get_pd_cache_high_priority_max_age_seconds.cache_clear()

    assert get_pd_cache_high_priority_max_age_seconds() == 30

    get_pd_cache_high_priority_max_age_seconds.cache_clear()


def test_pd_cache_high_priority_min_prompt_tokens_defaults_to_4096(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_PD_CACHE_HIGH_PRIORITY_MIN_PROMPT_TOKENS", raising=False)
    get_pd_cache_high_priority_min_prompt_tokens.cache_clear()

    assert get_pd_cache_high_priority_min_prompt_tokens() == 4096

    get_pd_cache_high_priority_min_prompt_tokens.cache_clear()


def test_pd_cache_high_priority_min_prompt_tokens_reads_environment_variable(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_PD_CACHE_HIGH_PRIORITY_MIN_PROMPT_TOKENS", "2048")
    get_pd_cache_high_priority_min_prompt_tokens.cache_clear()

    assert get_pd_cache_high_priority_min_prompt_tokens() == 2048

    get_pd_cache_high_priority_min_prompt_tokens.cache_clear()


def test_pd_node_resource_wait_timeout_defaults_to_10_seconds(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_PD_NODE_RESOURCE_WAIT_TIMEOUT_SECONDS", raising=False)
    get_pd_node_resource_wait_timeout_seconds.cache_clear()

    assert get_pd_node_resource_wait_timeout_seconds() == 10

    get_pd_node_resource_wait_timeout_seconds.cache_clear()


def test_pd_node_resource_wait_timeout_reads_environment_variable(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_PD_NODE_RESOURCE_WAIT_TIMEOUT_SECONDS", "30")
    get_pd_node_resource_wait_timeout_seconds.cache_clear()

    assert get_pd_node_resource_wait_timeout_seconds() == 30

    get_pd_node_resource_wait_timeout_seconds.cache_clear()


def test_pd_node_continuation_resource_wait_timeout_defaults_to_60_seconds(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_PD_NODE_CONTINUATION_RESOURCE_WAIT_TIMEOUT_SECONDS", raising=False)
    get_pd_node_continuation_resource_wait_timeout_seconds.cache_clear()

    assert get_pd_node_continuation_resource_wait_timeout_seconds() == 60

    get_pd_node_continuation_resource_wait_timeout_seconds.cache_clear()


def test_pd_node_continuation_resource_wait_timeout_reads_environment_variable(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_PD_NODE_CONTINUATION_RESOURCE_WAIT_TIMEOUT_SECONDS", "90")
    get_pd_node_continuation_resource_wait_timeout_seconds.cache_clear()

    assert get_pd_node_continuation_resource_wait_timeout_seconds() == 90

    get_pd_node_continuation_resource_wait_timeout_seconds.cache_clear()


def test_pd_node_busy_retry_timeout_defaults_to_120_seconds(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_PD_NODE_BUSY_RETRY_TIMEOUT_SECONDS", raising=False)
    get_pd_node_busy_retry_timeout_seconds.cache_clear()

    assert get_pd_node_busy_retry_timeout_seconds() == 120

    get_pd_node_busy_retry_timeout_seconds.cache_clear()


def test_pd_node_busy_retry_timeout_reads_environment_variable(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_PD_NODE_BUSY_RETRY_TIMEOUT_SECONDS", "60")
    get_pd_node_busy_retry_timeout_seconds.cache_clear()

    assert get_pd_node_busy_retry_timeout_seconds() == 60

    get_pd_node_busy_retry_timeout_seconds.cache_clear()
