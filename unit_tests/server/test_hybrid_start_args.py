import json

import pytest

from lightllm.server.api_start import _launch_subprocesses
from lightllm.server.core.objs.start_args_type import StartArgs


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


def test_normal_internal_calibration_allows_disabled_dynamic_prompt_cache(launch_args):
    args = launch_args(
        "qwen3_5_text",
        run_mode="normal",
        export_fp8kv_calibration=True,
        calibration_target="qkv",
        llm_kv_type="None",
        disable_dynamic_prompt_cache=True,
        disable_cudagraph=True,
    )
    with pytest.raises(RuntimeError, match="startup validation passed"):
        _launch_subprocesses(args)


@pytest.mark.parametrize("run_mode", ["prefill", "decode"])
def test_non_normal_internal_calibration_keeps_hybrid_dynamic_cache_requirement(launch_args, run_mode):
    args = launch_args(
        "qwen3_5_text",
        run_mode=run_mode,
        export_fp8kv_calibration=True,
        calibration_target="qkv",
        llm_kv_type="None",
        disable_dynamic_prompt_cache=True,
    )
    with pytest.raises(AssertionError, match="hybrid attention models require dynamic prompt cache"):
        _launch_subprocesses(args)


def test_internal_calibration_keeps_hybrid_chunked_prefill_requirement(launch_args):
    args = launch_args(
        "qwen3_5_text",
        run_mode="normal",
        export_fp8kv_calibration=True,
        calibration_target="qkv",
        llm_kv_type="None",
        disable_dynamic_prompt_cache=True,
        disable_cudagraph=True,
        disable_chunked_prefill=True,
    )
    with pytest.raises(AssertionError, match="hybrid attention models require chunked prefill"):
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
