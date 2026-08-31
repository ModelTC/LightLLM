from easydict import EasyDict

from lightllm.utils import envs_utils


def _set_start_args(monkeypatch, **kwargs):
    args = EasyDict(
        graph_max_batch_size=kwargs.get("graph_max_batch_size", 256),
        running_max_req_size=kwargs.get("running_max_req_size", 256),
        mtp_mode=kwargs.get("mtp_mode"),
        mtp_step=kwargs.get("mtp_step", 0),
        enable_tpsp_mix_mode=kwargs.get("enable_tpsp_mix_mode", False),
        enable_ep_moe=kwargs.get("enable_ep_moe", False),
        tp=kwargs.get("tp", 1),
    )
    monkeypatch.setattr(envs_utils, "get_env_start_args", lambda: args)
    envs_utils.get_deepep_num_max_dispatch_tokens_per_rank_decode.cache_clear()


def test_deepep_decode_limit_covers_speculative_physical_batch(monkeypatch):
    monkeypatch.delenv("NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE", raising=False)
    _set_start_args(
        monkeypatch,
        graph_max_batch_size=64,
        running_max_req_size=64,
        mtp_mode="eagle_with_att",
        mtp_step=5,
    )

    assert envs_utils.get_deepep_num_max_dispatch_tokens_per_rank_decode() == 384


def test_deepep_decode_limit_uses_sequence_parallel_local_batch(monkeypatch):
    monkeypatch.delenv("NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE", raising=False)
    _set_start_args(
        monkeypatch,
        graph_max_batch_size=64,
        running_max_req_size=64,
        mtp_mode="eagle_with_att",
        mtp_step=5,
        enable_tpsp_mix_mode=True,
        enable_ep_moe=True,
        tp=8,
    )

    assert envs_utils.get_deepep_num_max_dispatch_tokens_per_rank_decode() == 256


def test_deepep_decode_limit_is_aligned_and_keeps_minimum(monkeypatch):
    monkeypatch.delenv("NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE", raising=False)
    _set_start_args(
        monkeypatch,
        graph_max_batch_size=63,
        running_max_req_size=32,
        mtp_mode="eagle_with_att",
        mtp_step=4,
    )
    assert envs_utils.get_deepep_num_max_dispatch_tokens_per_rank_decode() == 320

    _set_start_args(monkeypatch, graph_max_batch_size=8, running_max_req_size=8)
    assert envs_utils.get_deepep_num_max_dispatch_tokens_per_rank_decode() == 256


def test_deepep_decode_limit_honors_explicit_override(monkeypatch):
    monkeypatch.setenv("NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE", "512")
    _set_start_args(
        monkeypatch,
        graph_max_batch_size=64,
        running_max_req_size=64,
        mtp_mode="eagle_with_att",
        mtp_step=5,
    )

    assert envs_utils.get_deepep_num_max_dispatch_tokens_per_rank_decode() == 512
