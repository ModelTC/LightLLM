from types import SimpleNamespace

from lightllm.utils import envs_utils
from lightllm.utils.envs_utils import (
    get_deepep_num_max_dispatch_tokens_per_rank,
    get_deepep_num_max_dispatch_tokens_per_rank_decode,
    get_deepep_num_max_dispatch_tokens_per_rank_prefill,
    is_stop_matching_in_reasoning_enabled,
    get_pd_cache_high_priority_max_age_seconds,
    get_pd_cache_high_priority_min_prompt_tokens,
    get_pd_node_busy_retry_timeout_seconds,
    get_pd_node_continuation_resource_wait_timeout_seconds,
    get_pd_node_resource_wait_timeout_seconds,
)


def test_stop_in_reasoning_defaults_to_false(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_STOP_IN_REASONING", raising=False)
    is_stop_matching_in_reasoning_enabled.cache_clear()
    assert is_stop_matching_in_reasoning_enabled() is False
    is_stop_matching_in_reasoning_enabled.cache_clear()


def test_stop_in_reasoning_reads_environment(monkeypatch):
    for value, expected in [("1", True), ("true", True), ("ON", True), ("0", False), ("false", False)]:
        monkeypatch.setenv("LIGHTLLM_STOP_IN_REASONING", value)
        is_stop_matching_in_reasoning_enabled.cache_clear()
        assert is_stop_matching_in_reasoning_enabled() is expected
    is_stop_matching_in_reasoning_enabled.cache_clear()


def test_pd_cache_high_priority_max_age_defaults_to_180_seconds(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_PD_CACHE_HIGH_PRIORITY_MAX_AGE_SECONDS", raising=False)
    get_pd_cache_high_priority_max_age_seconds.cache_clear()

    assert get_pd_cache_high_priority_max_age_seconds() == 180

    get_pd_cache_high_priority_max_age_seconds.cache_clear()


def test_pd_cache_high_priority_max_age_reads_environment_variable(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_PD_CACHE_HIGH_PRIORITY_MAX_AGE_SECONDS", "30")
    get_pd_cache_high_priority_max_age_seconds.cache_clear()

    assert get_pd_cache_high_priority_max_age_seconds() == 30

    get_pd_cache_high_priority_max_age_seconds.cache_clear()


def test_pd_cache_high_priority_min_prompt_tokens_defaults_to_2048(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_PD_CACHE_HIGH_PRIORITY_MIN_PROMPT_TOKENS", raising=False)
    get_pd_cache_high_priority_min_prompt_tokens.cache_clear()

    assert get_pd_cache_high_priority_min_prompt_tokens() == 2048

    get_pd_cache_high_priority_min_prompt_tokens.cache_clear()


def test_pd_cache_high_priority_min_prompt_tokens_reads_environment_variable(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_PD_CACHE_HIGH_PRIORITY_MIN_PROMPT_TOKENS", "2048")
    get_pd_cache_high_priority_min_prompt_tokens.cache_clear()

    assert get_pd_cache_high_priority_min_prompt_tokens() == 2048

    get_pd_cache_high_priority_min_prompt_tokens.cache_clear()


def test_pd_node_resource_wait_timeout_defaults_to_20_seconds(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_PD_NODE_RESOURCE_WAIT_TIMEOUT_SECONDS", raising=False)
    get_pd_node_resource_wait_timeout_seconds.cache_clear()

    assert get_pd_node_resource_wait_timeout_seconds() == 20

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


def test_pd_node_busy_retry_timeout_defaults_to_zero(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_PD_NODE_BUSY_RETRY_TIMEOUT_SECONDS", raising=False)
    get_pd_node_busy_retry_timeout_seconds.cache_clear()

    assert get_pd_node_busy_retry_timeout_seconds() == 0

    get_pd_node_busy_retry_timeout_seconds.cache_clear()


def test_pd_node_busy_retry_timeout_reads_environment_variable(monkeypatch):
    monkeypatch.setenv("LIGHTLLM_PD_NODE_BUSY_RETRY_TIMEOUT_SECONDS", "60")
    get_pd_node_busy_retry_timeout_seconds.cache_clear()

    assert get_pd_node_busy_retry_timeout_seconds() == 60

    get_pd_node_busy_retry_timeout_seconds.cache_clear()


def test_deepep_decode_buffer_capacity_covers_mtp_rows_and_safety_margin(monkeypatch):
    monkeypatch.delenv("NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE", raising=False)
    monkeypatch.setattr(
        envs_utils,
        "get_env_start_args",
        lambda: SimpleNamespace(running_max_req_size=256, mtp_step=2),
    )
    get_deepep_num_max_dispatch_tokens_per_rank_decode.cache_clear()

    assert get_deepep_num_max_dispatch_tokens_per_rank_decode() == 784

    get_deepep_num_max_dispatch_tokens_per_rank_decode.cache_clear()


def test_deepep_prefill_buffer_capacity_adds_safety_margin(monkeypatch):
    monkeypatch.delenv("NUM_MAX_DISPATCH_TOKENS_PER_RANK_PREFILL", raising=False)
    monkeypatch.setattr(envs_utils, "get_env_start_args", lambda: SimpleNamespace(batch_max_tokens=6000))
    get_deepep_num_max_dispatch_tokens_per_rank_prefill.cache_clear()

    assert get_deepep_num_max_dispatch_tokens_per_rank_prefill() == 6128

    get_deepep_num_max_dispatch_tokens_per_rank_prefill.cache_clear()


def test_deepep_buffer_capacity_uses_larger_phase_capacity(monkeypatch):
    cases = [
        (4096, 256, 4096),
        (256, 512, 512),
    ]

    for prefill_capacity, decode_capacity, expected_capacity in cases:
        monkeypatch.setattr(
            envs_utils,
            "get_deepep_num_max_dispatch_tokens_per_rank_prefill",
            lambda prefill_capacity=prefill_capacity: prefill_capacity,
        )
        monkeypatch.setattr(
            envs_utils,
            "get_deepep_num_max_dispatch_tokens_per_rank_decode",
            lambda decode_capacity=decode_capacity: decode_capacity,
        )
        get_deepep_num_max_dispatch_tokens_per_rank.cache_clear()
        assert get_deepep_num_max_dispatch_tokens_per_rank() == expected_capacity

    get_deepep_num_max_dispatch_tokens_per_rank.cache_clear()


def test_deepep_buffer_capacity_keeps_explicit_environment_override(monkeypatch):
    monkeypatch.setenv("NUM_MAX_DISPATCH_TOKENS_PER_RANK_PREFILL", "1024")
    monkeypatch.setenv("NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE", "128")
    get_deepep_num_max_dispatch_tokens_per_rank_prefill.cache_clear()
    get_deepep_num_max_dispatch_tokens_per_rank_decode.cache_clear()

    assert get_deepep_num_max_dispatch_tokens_per_rank_prefill() == 1024
    assert get_deepep_num_max_dispatch_tokens_per_rank_decode() == 128

    get_deepep_num_max_dispatch_tokens_per_rank_prefill.cache_clear()
    get_deepep_num_max_dispatch_tokens_per_rank_decode.cache_clear()
