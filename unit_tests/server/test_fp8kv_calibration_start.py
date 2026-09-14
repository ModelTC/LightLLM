import pytest

from lightllm.server.api_start import _launch_subprocesses
from lightllm.server.core.objs.start_args_type import StartArgs


def test_export_calibration_rejects_prefill_cuda_graph(monkeypatch):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)
    args = StartArgs(
        export_fp8kv_calibration=True,
        disable_cudagraph=True,
        enable_prefill_cudagraph=True,
    )
    with pytest.raises(AssertionError, match="prefill CUDA Graph"):
        _launch_subprocesses(args)


def test_dspark_normal_export_calibration_allows_disable_cudagraph(monkeypatch):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)

    class ReachedAutoConfig(Exception):
        pass

    def reached_auto_config(args):
        raise ReachedAutoConfig

    monkeypatch.setattr("lightllm.server.api_start.auto_set_max_req_total_len", reached_auto_config)
    args = StartArgs(
        run_mode="normal",
        mtp_mode="dspark",
        mtp_draft_model_dir=["draft"],
        mtp_step=2,
        export_fp8kv_calibration=True,
        disable_cudagraph=True,
    )

    with pytest.raises(ReachedAutoConfig):
        _launch_subprocesses(args)


def test_dspark_normal_without_export_still_rejects_disable_cudagraph(monkeypatch):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)
    args = StartArgs(
        run_mode="normal",
        mtp_mode="dspark",
        mtp_draft_model_dir=["draft"],
        mtp_step=2,
        disable_cudagraph=True,
    )

    with pytest.raises(AssertionError, match="only supported on Prefill nodes"):
        _launch_subprocesses(args)


def test_dspark_export_calibration_rejects_prefill_cuda_graph(monkeypatch):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.auto_set_max_req_total_len", lambda args: None)
    args = StartArgs(
        run_mode="normal",
        mtp_mode="dspark",
        mtp_draft_model_dir=["draft"],
        mtp_step=2,
        export_fp8kv_calibration=True,
        disable_cudagraph=True,
        enable_prefill_cudagraph=True,
    )

    with pytest.raises(AssertionError, match="prefill CUDA Graph"):
        _launch_subprocesses(args)
