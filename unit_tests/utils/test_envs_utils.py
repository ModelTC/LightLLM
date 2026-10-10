from lightllm.utils.envs_utils import (
    get_stop_in_reasoning,
    get_pd_cache_high_priority_max_age_seconds,
    get_pd_cache_high_priority_min_prompt_tokens,
    get_pd_node_busy_retry_timeout_seconds,
    get_pd_node_continuation_resource_wait_timeout_seconds,
    get_pd_node_resource_wait_timeout_seconds,
)


def test_stop_in_reasoning_defaults_to_false(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_STOP_IN_REASONING", raising=False)
    get_stop_in_reasoning.cache_clear()
    assert get_stop_in_reasoning() is False
    get_stop_in_reasoning.cache_clear()


def test_stop_in_reasoning_reads_environment(monkeypatch):
    for value, expected in [("1", True), ("true", True), ("ON", True), ("0", False), ("false", False)]:
        monkeypatch.setenv("LIGHTLLM_STOP_IN_REASONING", value)
        get_stop_in_reasoning.cache_clear()
        assert get_stop_in_reasoning() is expected
    get_stop_in_reasoning.cache_clear()


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
