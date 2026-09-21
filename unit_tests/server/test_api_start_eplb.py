import pytest

from lightllm.server import api_start
from lightllm.server.core.objs.start_args_type import StartArgs


def test_eplb_redundant_expert_count_must_not_be_negative(monkeypatch):
    args = StartArgs(
        eplb_num_redundant_experts_per_rank=-1,
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
    )

    monkeypatch.setattr(api_start, "_set_envs_and_config", lambda args: None)
    monkeypatch.setattr(api_start, "auto_set_max_req_total_len", lambda args: None)
    monkeypatch.setattr(api_start, "auto_set_fused_shared_experts", lambda args: None)
    monkeypatch.setattr(api_start, "set_unique_server_name", lambda args: None)

    with pytest.raises(
        AssertionError,
        match="--eplb_num_redundant_experts_per_rank must be greater than or equal to 0",
    ):
        api_start._launch_subprocesses(args)


def test_eplb_rebalance_count_must_not_be_less_than_negative_one(monkeypatch):
    args = StartArgs(
        eplb_rebalance_count=-2,
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
    )

    monkeypatch.setattr(api_start, "_set_envs_and_config", lambda args: None)
    monkeypatch.setattr(api_start, "auto_set_max_req_total_len", lambda args: None)
    monkeypatch.setattr(api_start, "auto_set_fused_shared_experts", lambda args: None)
    monkeypatch.setattr(api_start, "set_unique_server_name", lambda args: None)

    with pytest.raises(
        AssertionError,
        match="--eplb_rebalance_count must be greater than or equal to -1",
    ):
        api_start._launch_subprocesses(args)


def test_eplb_redundant_experts_require_ep_moe(monkeypatch):
    args = StartArgs(
        enable_ep_moe=False,
        eplb_num_redundant_experts_per_rank=1,
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
    )

    monkeypatch.setattr(api_start, "_set_envs_and_config", lambda args: None)
    monkeypatch.setattr(api_start, "auto_set_max_req_total_len", lambda args: None)
    monkeypatch.setattr(api_start, "auto_set_fused_shared_experts", lambda args: None)
    monkeypatch.setattr(api_start, "set_unique_server_name", lambda args: None)

    with pytest.raises(AssertionError, match="EPLB requires --enable_ep_moe"):
        api_start._launch_subprocesses(args)


def test_eplb_prefill_cudagraph_is_rejected_before_starting_subprocesses(monkeypatch):
    args = StartArgs(
        enable_ep_moe=True,
        eplb_num_redundant_experts_per_rank=2,
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

    with pytest.raises(AssertionError, match="EPLB does not support --enable_prefill_cudagraph"):
        api_start._launch_subprocesses(args)


def test_eplb_redundant_experts_cannot_be_combined_with_rl(monkeypatch):
    args = StartArgs(
        enable_ep_moe=True,
        enable_rl=True,
        eplb_num_redundant_experts_per_rank=2,
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

    with pytest.raises(AssertionError, match="EPLB redundant experts do not support --enable_rl"):
        api_start._launch_subprocesses(args)


def test_eplb_mtp_combination_is_not_rejected_before_starting_subprocesses(monkeypatch):
    args = StartArgs(
        model_dir="test-model",
        enable_ep_moe=True,
        eplb_num_redundant_experts_per_rank=2,
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
    monkeypatch.setattr(api_start, "is_sm100_gpu", lambda: False)
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
    monkeypatch.setattr(api_start.process_manager, "setup_exit_controller", lambda: None)
    monkeypatch.setattr(api_start.process_manager, "register_process_tree", lambda process: None)

    api_start._launch_subprocesses(args)
