import json

import pytest

from lightllm.server import api_start
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.fixture
def nvfp4_launch_args(monkeypatch, tmp_path):
    for name in (
        "_set_envs_and_config",
        "auto_set_max_req_total_len",
        "auto_set_fused_shared_experts",
        "set_unique_server_name",
        "auto_set_response_parsers",
        "auto_configure_allreduce_flags_from_args",
    ):
        monkeypatch.setattr(api_start, name, lambda args: None)

    def validation_finished(ports):
        raise RuntimeError("startup configuration reached")

    monkeypatch.setattr(api_start, "validate_ports", validation_finished)

    def make_args(checkpoint_dtype="bfloat16", **kwargs):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama", "torch_dtype": checkpoint_dtype}))
        return StartArgs(
            model_dir=str(tmp_path),
            llm_kv_type="nvfp4kv",
            max_req_total_len=8192,
            eos_id=[2],
            disable_vision=True,
            disable_audio=True,
            disable_shm_warning=True,
            **kwargs,
        )

    return make_args


@pytest.mark.parametrize("checkpoint_dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("data_type", [None, "fp16", "float16", "bf16", "bfloat16"])
def test_nvfp4_startup_resolves_default_dtype_and_accepts_aliases(nvfp4_launch_args, checkpoint_dtype, data_type):
    args = nvfp4_launch_args(checkpoint_dtype, data_type=data_type)
    with pytest.raises(RuntimeError, match="startup configuration reached"):
        api_start._launch_subprocesses(args)
    assert args.data_type == (checkpoint_dtype if data_type is None else data_type)


@pytest.mark.parametrize("data_type", [None, "fp32", "float32"])
def test_nvfp4_startup_rejects_resolved_float32(nvfp4_launch_args, data_type):
    args = nvfp4_launch_args("float32", data_type=data_type)
    with pytest.raises(AssertionError, match="NVFP4 requires float16 or bfloat16"):
        api_start._launch_subprocesses(args)


def test_nvfp4_startup_rejects_static_calibration(nvfp4_launch_args):
    args = nvfp4_launch_args(kv_quant_calibration_config_path="unused.json")
    with pytest.raises(AssertionError, match="NVFP4 uses dynamic blocks and unit global scales"):
        api_start._launch_subprocesses(args)
