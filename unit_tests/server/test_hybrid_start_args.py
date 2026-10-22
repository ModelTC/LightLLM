import json

import pytest

from lightllm.server.api_start import _launch_subprocesses
from lightllm.server.core.objs.start_args_type import StartArgs

# These tests check the cuda start-up validation, the xpu start policy would reject their modes first.
pytestmark = pytest.mark.usefixtures("cuda_platform")


@pytest.fixture
def launch_args(monkeypatch, tmp_path):
    for name in (
        "_set_envs_and_config",
        "auto_set_max_req_total_len",
        "auto_set_fused_shared_experts",
        "set_unique_server_name",
    ):
        monkeypatch.setattr(f"lightllm.server.api_start.{name}", lambda args: None)

    def validation_finished(args):
        raise RuntimeError("startup validation passed")

    monkeypatch.setattr("lightllm.server.api_start.auto_set_response_parsers", validation_finished)

    def make_args(model_type, **kwargs):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
        return StartArgs(
            model_dir=str(tmp_path),
            max_req_total_len=2048,
            eos_id=[2],
            router_token_ratio=0.0,
            disable_vision=True,
            disable_audio=True,
            disable_shm_warning=True,
            **kwargs,
        )

    return make_args


@pytest.mark.parametrize("run_mode", ["normal", "prefill", "decode"])
@pytest.mark.parametrize("output_constraint_mode", ["none", "xgrammar"])
@pytest.mark.parametrize(
    "options, message",
    [
        ({"disable_dynamic_prompt_cache": True}, "hybrid attention models require dynamic prompt cache"),
        ({"disable_chunked_prefill": True}, "hybrid attention models require chunked prefill"),
        ({"diverse_mode": True}, "--diverse_mode does not support hybrid attention models"),
    ],
)
def test_hybrid_rejects_unsupported_modes(launch_args, run_mode, output_constraint_mode, options, message):
    args = launch_args("qwen3_5_text", run_mode=run_mode, output_constraint_mode=output_constraint_mode, **options)
    with pytest.raises(AssertionError, match=message):
        _launch_subprocesses(args)


@pytest.mark.parametrize("output_constraint_mode", ["none", "xgrammar"])
@pytest.mark.parametrize(
    "model_type, options",
    [
        ("qwen3_5_text", {}),
        ("qwen3", {"disable_dynamic_prompt_cache": True}),
        ("qwen3", {"disable_chunked_prefill": True}),
        ("qwen3", {"diverse_mode": True}),
    ],
)
def test_supported_models_pass_startup_validation(launch_args, output_constraint_mode, model_type, options):
    args = launch_args(model_type, output_constraint_mode=output_constraint_mode, **options)
    with pytest.raises(RuntimeError, match="startup validation passed"):
        _launch_subprocesses(args)


@pytest.mark.parametrize(
    "model_type,disable_mm,disable_vision,batch_tokens,passes",
    [
        ("qwen3", True, False, 512, False),
        ("qwen3", True, False, 1024, True),
        ("qwen3", True, False, 2048, True),
        ("qwen3", True, False, None, True),
        ("qwen3", False, False, 512, True),
        ("qwen3", True, True, 512, True),
        ("neo_chat", False, False, 512, False),
        ("deepseek_v4", False, False, 512, False),
        ("gemma4", False, False, 512, False),
        ("neo_chat", False, False, 1024, True),
        ("qwen3_vl", False, False, 512, True),
        ("qwen3_5", False, False, 512, True),
    ],
)
def test_whole_image_blocks_fit_batch_at_startup(
    launch_args, model_type, disable_mm, disable_vision, batch_tokens, passes
):
    args = launch_args(
        model_type,
        disable_chunked_mm_input=disable_mm,
        batch_max_tokens=batch_tokens,
        max_image_token_count=1024,
    )
    args.disable_vision = disable_vision
    if passes:
        with pytest.raises(RuntimeError, match="startup validation passed"):
            _launch_subprocesses(args)
    else:
        with pytest.raises(AssertionError, match="batch_max_tokens >= max_image_token_count"):
            _launch_subprocesses(args)
