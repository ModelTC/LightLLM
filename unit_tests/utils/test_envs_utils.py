from types import SimpleNamespace

from lightllm.utils import envs_utils
from lightllm.utils.envs_utils import (
    get_deepep_num_max_dispatch_tokens_per_rank,
    get_deepep_num_max_dispatch_tokens_per_rank_decode,
    get_deepep_num_max_dispatch_tokens_per_rank_prefill,
)


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
