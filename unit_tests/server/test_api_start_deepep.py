import json

import pytest

from lightllm.server.api_start import _launch_subprocesses
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize("linear_backend", ["triton", "flashinfer"])
def test_decode_node_batch_max_tokens_covers_mtp_rows(monkeypatch, tmp_path, linear_backend):
    for name in (
        "_set_envs_and_config",
        "auto_set_max_req_total_len",
        "auto_set_fused_shared_experts",
        "set_unique_server_name",
        "auto_set_page_size",
    ):
        monkeypatch.setattr(f"lightllm.server.api_start.{name}", lambda args: None)

    def stop(args):
        raise RuntimeError("startup validation passed")

    monkeypatch.setattr("lightllm.server.api_start.auto_set_response_parsers", stop)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    args = StartArgs(
        run_mode="decode",
        model_dir=str(tmp_path),
        enable_ep_moe=True,
        llm_prefill_att_backend=["triton", linear_backend],
        running_max_req_size=256,
        mtp_mode="vanilla_no_att",
        mtp_step=2,
        max_req_total_len=4096,
        batch_max_tokens=4096,
        chunked_prefill_size=2048,
        eos_id=[0],
        data_type="float16",
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
        router_token_ratio=0.0,
    )

    with pytest.raises(RuntimeError, match="startup validation passed"):
        _launch_subprocesses(args)

    assert args.batch_max_tokens == 768


@pytest.mark.parametrize(
    "model_type,kv_type,requires_calibration",
    [
        ("qwen4_exp", "fp8kv_spt", False),
        ("qwen4_exp_text", "fp8kv_spt", False),
        ("qwen4_exp", "fp8kv_sph", True),
        ("llama", "fp8kv_spt", True),
    ],
)
def test_qsa_unit_scale_fp8_startup(monkeypatch, tmp_path, model_type, kv_type, requires_calibration):
    for name in (
        "_set_envs_and_config",
        "auto_set_max_req_total_len",
        "auto_set_fused_shared_experts",
        "set_unique_server_name",
        "auto_set_page_size",
    ):
        monkeypatch.setattr(f"lightllm.server.api_start.{name}", lambda args: None)

    def stop(args):
        raise RuntimeError("startup validation passed")

    monkeypatch.setattr("lightllm.server.api_start.auto_set_response_parsers", stop)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    args = StartArgs(
        model_dir=str(tmp_path),
        llm_kv_type=kv_type,
        running_max_req_size=256,
        max_req_total_len=4096,
        batch_max_tokens=4096,
        chunked_prefill_size=2048,
        eos_id=[0],
        data_type="bfloat16",
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
    )
    if requires_calibration:
        with pytest.raises(AssertionError, match="requires --kv_quant_calibration_config_path"):
            _launch_subprocesses(args)
    else:
        with pytest.raises(RuntimeError, match="startup validation passed"):
            _launch_subprocesses(args)
