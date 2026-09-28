import pytest

from lightllm.server.api_start import _launch_subprocesses
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize(
    "page_size,hash_page_size,cpu_page_size", [(1, 256, None), (256, 512, None), (256, 256, 4096), (256, 256, None)]
)
def test_dsv4_page_and_checkpoint_configuration(monkeypatch, page_size, hash_page_size, cpu_page_size):
    for name in (
        "_set_envs_and_config",
        "auto_set_max_req_total_len",
        "auto_set_fused_shared_experts",
        "set_unique_server_name",
    ):
        monkeypatch.setattr("lightllm.server.api_start." + name, lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.get_model_type", lambda model_dir: "deepseek_v4")
    monkeypatch.setattr("lightllm.server.api_start.is_hybrid_att_model", lambda model_dir: True)

    def validation_finished(args):
        assert args.cpu_cache_token_page_size == 2048
        raise RuntimeError("DSV4 page-size validation passed")

    monkeypatch.setattr("lightllm.server.api_start.auto_set_response_parsers", validation_finished)
    args = StartArgs(
        model_dir="unused",
        page_size=page_size,
        linear_att_hash_page_size=hash_page_size,
        linear_att_page_block_num=8,
        enable_cpu_cache=True,
        cpu_cache_token_page_size=cpu_page_size,
        max_req_total_len=8192,
        eos_id=[2],
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
    )
    if page_size != 256 or hash_page_size != 256:
        with pytest.raises(ValueError, match="DeepSeek-V4 requires"):
            _launch_subprocesses(args)
    elif cpu_page_size is not None:
        with pytest.raises(ValueError, match="CPU cache pages must match"):
            _launch_subprocesses(args)
    else:
        with pytest.raises(RuntimeError, match="DSV4 page-size validation passed"):
            _launch_subprocesses(args)


def test_pd_kv_page_size_must_be_divisible_by_model_page_size(monkeypatch):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.auto_set_max_req_total_len", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.auto_set_fused_shared_experts", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.set_unique_server_name", lambda args: None)
    args = StartArgs(
        run_mode="decode",
        page_size=4,
        pd_kv_page_size=6,
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
    )

    with pytest.raises(AssertionError, match="--pd_kv_page_size must be divisible by --page_size"):
        _launch_subprocesses(args)


def test_normal_mode_does_not_validate_pd_kv_page_size(monkeypatch):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.auto_set_max_req_total_len", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.auto_set_fused_shared_experts", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.set_unique_server_name", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.is_hybrid_att_model", lambda model_dir: False)

    def validation_finished(args):
        raise RuntimeError("PD page-size validation passed")

    monkeypatch.setattr("lightllm.server.api_start.auto_set_response_parsers", validation_finished)
    args = StartArgs(
        run_mode="normal",
        model_dir="unused",
        page_size=4,
        pd_kv_page_size=6,
        eos_id=[2],
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
    )

    with pytest.raises(RuntimeError, match="PD page-size validation passed"):
        _launch_subprocesses(args)


@pytest.mark.parametrize("llm_kv_type", ["fp8kv_sph", "fp8kv_spt"])
def test_fp8_kv_cache_allows_multi_token_model_pages(monkeypatch, llm_kv_type):
    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.auto_set_max_req_total_len", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.auto_set_fused_shared_experts", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.set_unique_server_name", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.is_hybrid_att_model", lambda model_dir: False)

    def validation_finished(args):
        raise RuntimeError("FP8 page-size validation passed")

    monkeypatch.setattr("lightllm.server.api_start.auto_set_response_parsers", validation_finished)
    args = StartArgs(
        llm_kv_type=llm_kv_type,
        kv_quant_calibration_config_path="unused",
        model_dir="unused",
        page_size=16,
        eos_id=[2],
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
    )

    with pytest.raises(RuntimeError, match="FP8 page-size validation passed"):
        _launch_subprocesses(args)
