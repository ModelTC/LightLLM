import pytest

from lightllm.server.api_start import _launch_subprocesses
from lightllm.server.core.objs.start_args_type import StartArgs


def test_dynamic_mtp_requires_cuda_graph(monkeypatch):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)
    args = StartArgs(mtp_mode="vanilla_no_att", mtp_dynamic_verify=True, disable_cudagraph=True)

    with pytest.raises(AssertionError, match="--mtp_dynamic_verify"):
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
