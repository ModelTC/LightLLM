import pytest

from lightllm.server.api_start import _launch_subprocesses
from lightllm.server.core.objs.start_args_type import StartArgs


def test_mtp_requires_cuda_graph(monkeypatch):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)
    args = StartArgs(mtp_mode="vanilla_no_att", disable_cudagraph=True)

    with pytest.raises(AssertionError, match="--disable_cudagraph is not supported"):
        _launch_subprocesses(args)
