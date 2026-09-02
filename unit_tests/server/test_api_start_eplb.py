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


def test_eplb_mtp_combination_is_not_rejected_before_starting_subprocesses(monkeypatch):
    args = StartArgs(
        model_dir="test-model",
        enable_ep_moe=True,
        enable_prefill_eplb=True,
        mtp_mode="vanilla_no_att",
        mtp_step=1,
        eos_id=0,
        data_type="float16",
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
    )

    monkeypatch.setattr(api_start, "_set_envs_and_config", lambda args: None)
    monkeypatch.setattr(api_start, "auto_set_max_req_total_len", lambda args: None)
    monkeypatch.setattr(api_start, "auto_set_fused_shared_experts", lambda args: None)
    monkeypatch.setattr(api_start, "set_unique_server_name", lambda args: None)
    monkeypatch.setattr(api_start, "get_model_type", lambda model_dir: "llama")
    monkeypatch.setattr(api_start, "auto_set_response_parsers", lambda args: None)
    monkeypatch.setattr(api_start, "auto_configure_allreduce_flags_from_args", lambda args: None)
    monkeypatch.setattr(api_start, "validate_ports", lambda ports: None)
    monkeypatch.setattr(api_start, "set_env_start_args", lambda args: None)
    monkeypatch.setattr(api_start, "get_shm_port_args", lambda create=False: None)
    monkeypatch.setattr(api_start, "send_and_receive_node_ip", lambda args: None)
    monkeypatch.setattr(
        api_start.process_manager,
        "start_submodule_processes",
        lambda *args, **kwargs: (object(), None),
    )
    monkeypatch.setattr(api_start.process_manager, "register_process_tree", lambda process: None)

    api_start._launch_subprocesses(args)
