import pytest

from lightllm.server import api_start
from lightllm.server.core.objs.start_args_type import StartArgs


def test_eplb_prefill_cudagraph_is_rejected_before_starting_subprocesses(monkeypatch):
    args = StartArgs(
        enable_ep_moe=True,
        enable_prefill_eplb=True,
        enable_prefill_cudagraph=True,
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
    )

    monkeypatch.setattr(api_start, "_set_envs_and_config", lambda args: None)
    monkeypatch.setattr(api_start, "auto_set_max_req_total_len", lambda args: None)
    monkeypatch.setattr(api_start, "auto_set_fused_shared_experts", lambda args: None)
    monkeypatch.setattr(api_start, "set_unique_server_name", lambda args: None)
    monkeypatch.setattr(
        api_start.process_manager,
        "start_submodule_processes",
        lambda *args, **kwargs: pytest.fail("subprocess startup must not be reached"),
    )

    with pytest.raises(AssertionError, match="--enable_prefill_eplb does not support --enable_prefill_cudagraph"):
        api_start._launch_subprocesses(args)
