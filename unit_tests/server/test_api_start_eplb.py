import pytest

from lightllm.server import api_start
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize(
    "mode,redundant,run_mode,message",
    [
        ("full", 0, "normal", "requires --run_mode prefill"),
        ("full", 0, "decode", "requires --run_mode prefill"),
        ("full", -1, "prefill", "nonnegative"),
        ("redundant", 0, "prefill", "greater than 0"),
        ("unknown", 1, "prefill", "invalid --eplb_placement_mode"),
    ],
)
def test_full_eplb_mode_rejected_before_any_process_starts(monkeypatch, mode, redundant, run_mode, message):
    args = StartArgs(
        enable_ep_moe=True,
        enable_prefill_eplb=True,
        eplb_placement_mode=mode,
        eplb_num_redundant_experts_per_rank=redundant,
        run_mode=run_mode,
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
    )
    for name in (
        "_set_envs_and_config",
        "auto_set_max_req_total_len",
        "auto_set_fused_shared_experts",
        "set_unique_server_name",
    ):
        monkeypatch.setattr(api_start, name, lambda args: None)
    monkeypatch.setattr(api_start, "is_sm100_gpu", lambda: False)
    monkeypatch.setattr(
        api_start.process_manager,
        "start_submodule_processes",
        lambda *a, **kw: pytest.fail("must reject before process startup"),
    )
    with pytest.raises(AssertionError, match=message):
        api_start._launch_subprocesses(args)


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


@pytest.mark.parametrize("placement_mode,redundant,run_mode", [("redundant", 2, "normal"), ("full", 0, "prefill")])
def test_eplb_mtp_combination_is_not_rejected_before_starting_subprocesses(
    monkeypatch, placement_mode, redundant, run_mode
):
    args = StartArgs(
        model_dir="test-model",
        enable_ep_moe=True,
        enable_prefill_eplb=True,
        eplb_placement_mode=placement_mode,
        eplb_num_redundant_experts_per_rank=redundant,
        run_mode=run_mode,
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
    monkeypatch.setattr(api_start, "is_sm100_gpu", lambda: False)
    monkeypatch.setattr(
        api_start.process_manager,
        "start_submodule_processes",
        lambda *args, **kwargs: (object(), None),
    )
    monkeypatch.setattr(api_start.process_manager, "register_process_tree", lambda process: None)

    api_start._launch_subprocesses(args)
