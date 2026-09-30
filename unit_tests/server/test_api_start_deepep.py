from lightllm.server import api_start
from lightllm.server.core.objs.start_args_type import StartArgs


def test_decode_node_batch_max_tokens_covers_mtp_rows(monkeypatch):
    args = StartArgs(
        run_mode="decode",
        model_dir="test-model",
        enable_ep_moe=True,
        running_max_req_size=256,
        mtp_mode="vanilla_no_att",
        mtp_step=2,
        max_req_total_len=4096,
        batch_max_tokens=4096,
        chunked_prefill_size=2048,
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

    assert args.batch_max_tokens == 768
