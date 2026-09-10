import pytest

from lightllm.server.api_start import _launch_subprocesses
from lightllm.server.core.objs.start_args_type import StartArgs


def test_mtp_requires_cuda_graph(monkeypatch):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)
    args = StartArgs(mtp_mode="vanilla_no_att", disable_cudagraph=True)

    with pytest.raises(AssertionError, match="only supported on Prefill nodes"):
        _launch_subprocesses(args)


def test_mtp_prefill_still_requires_positive_step(monkeypatch):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.auto_set_max_req_total_len", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.auto_set_fused_shared_experts", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.set_unique_server_name", lambda args: None)
    args = StartArgs(
        run_mode="prefill",
        mtp_mode="dspark",
        mtp_draft_model_dir=["draft"],
        mtp_step=0,
        disable_cudagraph=True,
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
    )

    with pytest.raises(AssertionError):
        _launch_subprocesses(args)


def _patch_common_setup(monkeypatch):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.auto_set_max_req_total_len", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.auto_set_fused_shared_experts", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.set_unique_server_name", lambda args: None)


def test_mtp_asd_requires_mtp_mode(monkeypatch):
    _patch_common_setup(monkeypatch)
    args = StartArgs(
        run_mode="prefill",
        mtp_asd_regret_budget=2.0,
        disable_cudagraph=True,
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
    )

    with pytest.raises(AssertionError, match="requires an enabled mtp_mode"):
        _launch_subprocesses(args)


def test_mtp_asd_rejects_negative_budget(monkeypatch):
    _patch_common_setup(monkeypatch)
    args = StartArgs(
        run_mode="prefill",
        mtp_mode="dspark",
        mtp_draft_model_dir=["draft"],
        mtp_step=1,
        mtp_asd_regret_budget=-1.0,
        disable_cudagraph=True,
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
    )

    with pytest.raises(AssertionError, match="must be >= 0"):
        _launch_subprocesses(args)


def test_mtp_asd_rejects_negative_ratio_and_mismatch(monkeypatch):
    _patch_common_setup(monkeypatch)
    base_kwargs = dict(
        run_mode="prefill",
        mtp_mode="dspark",
        mtp_draft_model_dir=["draft"],
        mtp_step=1,
        mtp_asd_regret_budget=2.0,
        disable_cudagraph=True,
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
    )

    with pytest.raises(AssertionError, match="must be >= 0"):
        _launch_subprocesses(StartArgs(mtp_asd_local_regret_ratio=-0.1, **base_kwargs))
    with pytest.raises(AssertionError, match="must be >= 0"):
        _launch_subprocesses(StartArgs(mtp_asd_block_max_mismatch=-1, **base_kwargs))
