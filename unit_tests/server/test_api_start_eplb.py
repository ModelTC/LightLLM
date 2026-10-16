import pytest

from lightllm.common.eplb_utils import is_eplb_enabled
from lightllm.server import api_start
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils import device_utils


@pytest.fixture(autouse=True)
def supported_eplb_device(monkeypatch):
    monkeypatch.setattr(device_utils, "is_sm100_gpu", lambda: False)


@pytest.mark.parametrize(
    "redundant,run_mode,nnodes,error,message",
    [
        (0, "normal", 2, ValueError, "requires a single node"),
        (-1, "prefill", 1, AssertionError, "nonnegative"),
    ],
)
def test_full_eplb_mode_rejected_before_any_process_starts(monkeypatch, redundant, run_mode, nnodes, error, message):
    args = StartArgs(
        enable_ep_moe=True,
        enable_prefill_eplb=True,
        eplb_num_redundant_experts_per_rank=redundant,
        run_mode=run_mode,
        nnodes=nnodes,
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
    monkeypatch.setattr(
        api_start.process_manager,
        "start_submodule_processes",
        lambda *a, **kw: pytest.fail("must reject before process startup"),
    )
    with pytest.raises(error, match=message):
        api_start._launch_subprocesses(args)


@pytest.mark.parametrize(
    "enabled,run_mode,nnodes,expected",
    [
        (False, "prefill", 1, False),
        (False, "normal", 2, False),
        (False, "decode", 1, False),
        (True, "prefill", 1, True),
        (True, "prefill", 2, True),
        (True, "normal", 1, True),
        (True, "decode", 1, False),
        (True, "decode", 2, False),
    ],
)
def test_eplb_role_policy(enabled, run_mode, nnodes, expected):
    args = StartArgs(enable_ep_moe=True, enable_prefill_eplb=enabled, run_mode=run_mode, nnodes=nnodes)
    assert is_eplb_enabled(args) is expected


def test_eplb_rejects_unsupported_inference_role():
    with pytest.raises(ValueError, match="requires --run_mode prefill or single-node normal"):
        is_eplb_enabled(StartArgs(enable_prefill_eplb=True, run_mode="visual_only"))


@pytest.mark.parametrize("run_mode", ["prefill", "normal"])
@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"enable_ep_moe": False}, "requires --enable_ep_moe"),
        ({"enable_prefill_cudagraph": True}, "does not support --enable_prefill_cudagraph"),
        ({"enable_rl": True}, "does not support --enable_rl"),
        ({"eplb_num_redundant_experts_per_rank": -1}, "nonnegative"),
    ],
)
def test_eplb_policy_owns_all_configuration_rejections(run_mode, overrides, message):
    args = StartArgs(enable_ep_moe=True, enable_prefill_eplb=True, run_mode=run_mode)
    for name, value in overrides.items():
        setattr(args, name, value)
    with pytest.raises(AssertionError, match=message):
        is_eplb_enabled(args)


@pytest.mark.parametrize("run_mode", ["prefill", "normal"])
def test_eplb_policy_rejects_sm100(monkeypatch, run_mode):
    monkeypatch.setattr(device_utils, "is_sm100_gpu", lambda: True)
    args = StartArgs(enable_ep_moe=True, enable_prefill_eplb=True, run_mode=run_mode)
    with pytest.raises(AssertionError, match="does not support SM100"):
        is_eplb_enabled(args)


@pytest.mark.parametrize("enabled,run_mode", [(False, "normal"), (False, "visual_only"), (True, "decode")])
def test_inactive_eplb_skips_support_checks_and_device_queries(monkeypatch, enabled, run_mode):
    monkeypatch.setattr(device_utils, "is_sm100_gpu", lambda: pytest.fail("inactive EPLB must not query CUDA"))
    args = StartArgs(
        enable_prefill_eplb=enabled,
        run_mode=run_mode,
        nnodes=2,
        enable_ep_moe=False,
        enable_prefill_cudagraph=True,
        enable_rl=True,
        eplb_num_redundant_experts_per_rank=-1,
    )
    assert is_eplb_enabled(args) is False


def test_eplb_rl_is_rejected_before_starting_subprocesses(monkeypatch):
    args = StartArgs(
        enable_ep_moe=True,
        enable_prefill_eplb=True,
        enable_rl=True,
        run_mode="normal",
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
    with pytest.raises(AssertionError, match="does not support --enable_rl"):
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


@pytest.mark.parametrize("redundant", [0, 2])
@pytest.mark.parametrize(
    "run_mode,nnodes,ep,sm100,expected",
    [
        ("prefill", 1, True, False, True),
        ("prefill", 2, True, False, True),
        ("normal", 1, True, False, True),
        ("decode", 1, True, False, False),
        ("decode", 2, True, True, False),
        ("decode", 1, False, True, False),
    ],
)
def test_eplb_role_resolution_reaches_subprocesses_with_mtp(
    monkeypatch, redundant, run_mode, nnodes, ep, sm100, expected
):
    monkeypatch.setattr(api_start, "is_hybrid_att_model", lambda _: False)
    args = StartArgs(
        model_dir="test-model",
        enable_ep_moe=ep,
        enable_prefill_eplb=True,
        eplb_num_redundant_experts_per_rank=redundant,
        run_mode=run_mode,
        nnodes=nnodes,
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
    monkeypatch.setattr(device_utils, "is_sm100_gpu", lambda: sm100)
    monkeypatch.setattr(api_start, "auto_set_response_parsers", lambda args: None)
    monkeypatch.setattr(api_start, "auto_configure_allreduce_flags_from_args", lambda args: None)
    monkeypatch.setattr(api_start, "validate_ports", lambda ports: None)
    exported = []
    monkeypatch.setattr(api_start, "set_env_start_args", lambda args: exported.append(args.enable_prefill_eplb))
    monkeypatch.setattr(api_start, "get_shm_port_args", lambda create=False: None)
    monkeypatch.setattr(api_start, "send_and_receive_node_ip", lambda args: None)
    monkeypatch.setattr(api_start.process_manager, "setup_exit_controller", lambda: None)
    monkeypatch.setattr(
        api_start.process_manager,
        "start_submodule_processes",
        lambda *args, **kwargs: (object(), None),
    )
    monkeypatch.setattr(api_start.process_manager, "register_process_tree", lambda process: None)

    api_start._launch_subprocesses(args)
    assert args.enable_prefill_eplb is expected
    assert exported == [expected, expected]
