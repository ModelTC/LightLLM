import pytest

from lightllm.server import api_start
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize("mode", ["normal", "prefill", "decode"])
@pytest.mark.parametrize("dtype", [None, "bf16", "bfloat16"])
def test_megamoe_accepts_glm_modes(monkeypatch, mode, dtype):
    monkeypatch.setattr("lightllm.utils.device_utils.is_sm90_gpu", lambda: True)
    api_start._validate_ep_moe_backend(
        StartArgs(ep_moe_backend="megamoe", enable_ep_moe=True, run_mode=mode, data_type=dtype)
    )


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("enable_ep_moe", False, "requires --enable_ep_moe"),
        ("nnodes", 2, "single node"),
        ("ep_redundancy_expert_config_path", "redundant.json", "redundant experts"),
        ("auto_update_redundancy_expert", True, "redundant expert updates"),
        ("data_type", "fp16", "BF16"),
        ("enable_rl", True, "enable_rl"),
        ("enable_prefill_microbatch_overlap", True, "microbatch overlap"),
        ("enable_decode_microbatch_overlap", True, "microbatch overlap"),
    ],
)
def test_megamoe_rejects_unsupported_configuration(monkeypatch, field, value, error):
    monkeypatch.setattr("lightllm.utils.device_utils.is_sm90_gpu", lambda: True)
    args = StartArgs(ep_moe_backend="megamoe", enable_ep_moe=True)
    setattr(args, field, value)
    with pytest.raises(AssertionError, match=error):
        api_start._validate_ep_moe_backend(args)


def test_megamoe_requires_sm90(monkeypatch):
    monkeypatch.setattr("lightllm.utils.device_utils.is_sm90_gpu", lambda: False)
    with pytest.raises(AssertionError, match="SM90"):
        api_start._validate_ep_moe_backend(StartArgs(ep_moe_backend="megamoe", enable_ep_moe=True))
