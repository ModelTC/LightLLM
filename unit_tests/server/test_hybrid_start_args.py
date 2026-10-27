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


@pytest.mark.parametrize("mode,step", [("native", 0), ("compact", 4), ("replay", 0)])
def test_s3_configuration_is_prepared_before_ssm_validation(launch_args, monkeypatch, tmp_path, mode, step):
    from pathlib import Path
    from lightllm.utils.petrel_helper import PetrelHelper

    monkeypatch.chdir(tmp_path)
    model_dir = "s3://test-bucket/qwen"
    config = {"model_type": "qwen3_5_text"}
    downloads = []

    def list_files(path, extension):
        assert path == model_dir
        return ["config.json"] if extension == "json" else []

    def download_json(remote, local):
        downloads.append(remote)
        Path(local).write_text(json.dumps(config))

    monkeypatch.setattr(PetrelHelper, "list", list_files)
    monkeypatch.setattr(PetrelHelper, "download_json", download_json)
    args = launch_args("qwen3_5_text", ssm_state_mode=mode, mtp_step=step, mtp_mode="eagle_with_att" if step else None)
    args.model_dir = model_dir
    assert not Path(model_dir, "config.json").exists()
    with pytest.raises(RuntimeError, match="startup validation passed"):
        _launch_subprocesses(args)
    assert downloads == [model_dir + "/config.json"]
    if mode == "replay":
        assert args.ssm_replay_config == {"cache_len": 16, "compute_mode": "inline", "fold_method": "raw"}


def test_cold_s3_model_still_enforces_hybrid_constraints(launch_args, monkeypatch, tmp_path):
    from pathlib import Path

    monkeypatch.chdir(tmp_path)

    def prepare(model_dir):
        Path(model_dir).mkdir(parents=True)
        Path(model_dir, "config.json").write_text(json.dumps({"model_type": "qwen3_5_text"}))

    monkeypatch.setattr("lightllm.utils.petrel_helper.s3_model_prepare", prepare)
    args = launch_args("qwen3_5_text", disable_dynamic_prompt_cache=True)
    args.model_dir = "s3://test-bucket/qwen"
    with pytest.raises(AssertionError, match="hybrid attention models require dynamic prompt cache"):
        _launch_subprocesses(args)
