import pytest

from lightllm.server import api_start
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize("mode", ["normal", "prefill", "decode"])
@pytest.mark.parametrize("dtype", [None, "bf16", "bfloat16"])
def test_megamoe_accepts_glm_modes_and_bf16_aliases(monkeypatch, mode, dtype):
    monkeypatch.setattr(api_start, "is_sm90_gpu", lambda: True)
    api_start._validate_ep_moe_backend(
        StartArgs(ep_moe_backend="megamoe", enable_ep_moe=True, run_mode=mode, data_type=dtype)
    )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("enable_ep_moe", False, "requires --enable_ep_moe"),
        ("nnodes", 2, "single node"),
        ("enable_prefill_eplb", True, "enable_prefill_eplb"),
        ("data_type", "fp16", "BF16"),
        ("enable_rl", True, "enable_rl"),
        ("enable_prefill_microbatch_overlap", True, "microbatch overlap"),
        ("enable_decode_microbatch_overlap", True, "microbatch overlap"),
    ],
)
def test_megamoe_rejects_unsupported_configuration(monkeypatch, field, value, error):
    monkeypatch.setattr(api_start, "is_sm90_gpu", lambda: True)
    args = StartArgs(ep_moe_backend="megamoe", enable_ep_moe=True)
    setattr(args, field, value)
    with pytest.raises(AssertionError, match=error):
        api_start._validate_ep_moe_backend(args)


def test_megamoe_requires_sm90(monkeypatch):
    monkeypatch.setattr(api_start, "is_sm90_gpu", lambda: False)
    with pytest.raises(AssertionError, match="SM90"):
        api_start._validate_ep_moe_backend(StartArgs(ep_moe_backend="megamoe", enable_ep_moe=True))


def test_triton_remains_prefill_only(monkeypatch):
    monkeypatch.setattr(api_start, "is_sm90_gpu", lambda: True)
    with pytest.raises(AssertionError, match="prefill"):
        api_start._validate_ep_moe_backend(StartArgs(ep_moe_backend="triton", enable_ep_moe=True, run_mode="decode"))


@pytest.mark.parametrize("backend", ["auto", "megamoe"])
def test_ep_startup_accepts_flashkda(monkeypatch, backend):
    args = StartArgs(
        enable_ep_moe=True,
        ep_moe_backend=backend,
        llm_prefill_att_backend=["auto", "flashkda"],
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
        page_size=0,
    )
    for name in (
        "_set_envs_and_config",
        "auto_set_max_req_total_len",
        "auto_set_fused_shared_experts",
        "set_unique_server_name",
    ):
        monkeypatch.setattr(api_start, name, lambda args: None)
    monkeypatch.setattr(api_start, "get_model_type", lambda model_dir: "glm5_next")
    monkeypatch.setattr(api_start, "is_hybrid_att_model", lambda model_dir: True)
    monkeypatch.setattr(api_start, "is_sm90_gpu", lambda: True)
    monkeypatch.setattr(
        api_start.process_manager,
        "start_submodule_processes",
        lambda *args, **kwargs: pytest.fail("subprocess startup must not be reached"),
    )
    with pytest.raises(ValueError, match="--page_size"):
        api_start._launch_subprocesses(args)
