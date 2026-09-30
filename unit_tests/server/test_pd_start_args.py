import json

import pytest

from lightllm.server.api_start import _launch_subprocesses
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.fixture
def startup_args(monkeypatch):
    for name in (
        "_set_envs_and_config",
        "auto_set_max_req_total_len",
        "auto_set_fused_shared_experts",
        "set_unique_server_name",
    ):
        monkeypatch.setattr("lightllm.server.api_start." + name, lambda args: None)

    def validation_finished(args):
        raise RuntimeError("startup validation complete")

    monkeypatch.setattr("lightllm.server.api_start.auto_set_response_parsers", validation_finished)
    return StartArgs(
        model_dir="unused",
        max_req_total_len=8192,
        eos_id=[2],
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
    )


@pytest.mark.parametrize(
    "model_type,block_num,cpu_page_size,expected",
    [
        ("deepseek_v4", 8, None, 4096),
        ("deepseek_v4", 8, 4096, 4096),
        ("deepseek_v4", 10000000, None, 2048),
        ("deepseek_v4", 10000000, 4096, 4096),
        ("glm5_next", 8, None, 4096),
        ("glm5_next", 8, 256, 4096),
        ("llama", 8, None, 256),
        ("llama", 8, 1024, 1024),
    ],
)
def test_cpu_cache_page_defaults_preserve_model_contracts(
    startup_args, monkeypatch, model_type, block_num, cpu_page_size, expected
):
    monkeypatch.setattr("lightllm.server.api_start.get_model_type", lambda _: model_type)
    monkeypatch.setattr(
        "lightllm.server.api_start.is_hybrid_att_model", lambda _: model_type in ("deepseek_v4", "glm5_next")
    )
    args = startup_args
    args.enable_cpu_cache = True
    args.linear_att_page_block_num = block_num
    args.cpu_cache_token_page_size = cpu_page_size
    with pytest.raises(RuntimeError, match="startup validation complete"):
        _launch_subprocesses(args)
    assert args.cpu_cache_token_page_size == expected


@pytest.mark.parametrize("model_type", ["deepseek_v4", "llama"])
@pytest.mark.parametrize("enable_cpu_cache", [False, True])
def test_model_kv_defaults_preserve_hash_pages_kv_selection_and_explicit_pd_tuning(
    startup_args, monkeypatch, model_type, enable_cpu_cache
):
    monkeypatch.setattr("lightllm.server.api_start.get_model_type", lambda _: model_type)
    monkeypatch.setattr("lightllm.server.api_start.is_hybrid_att_model", lambda _: model_type == "deepseek_v4")
    args = startup_args
    args.enable_cpu_cache = enable_cpu_cache
    args.pd_kv_page_num, args.pd_kv_page_size = 8, 2048
    with pytest.raises(RuntimeError, match="startup validation complete"):
        _launch_subprocesses(args)
    assert args.page_size == (256 if model_type == "deepseek_v4" else 1)
    assert args.linear_att_hash_page_size == 512
    assert args.llm_kv_type == "None"
    assert (args.pd_kv_page_num, args.pd_kv_page_size) == (8, 2048)
    assert args.cache_placement_strategy == (
        "legacy" if enable_cpu_cache and model_type == "deepseek_v4" else "adaptive"
    )


def test_disabled_cpu_cache_keeps_page_size_unset(startup_args):
    with pytest.raises(RuntimeError, match="startup validation complete"):
        _launch_subprocesses(startup_args)
    assert startup_args.cpu_cache_token_page_size is None


def test_dsv4_cpu_cache_factory_resolves_packed_layout_without_overriding_kv_type(startup_args, monkeypatch, tmp_path):
    from lightllm.common.kv_cache_mem_manager import DeepseekV4MemoryManager
    from lightllm.common.kv_cache_mem_manager.mem_utils import select_mem_manager_class
    from lightllm.utils.envs_utils import get_env_start_args, get_added_mtp_kv_layer_num, get_llm_data_type
    from lightllm.utils.llm_utils import get_llm_model_class
    from lightllm.utils.kv_cache_utils import calcu_cpu_cache_meta

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "num_hidden_layers": 2,
                "head_dim": 512,
                "index_head_dim": 128,
                "compress_ratios": [4, 128],
            }
        )
    )
    args = startup_args
    args.model_dir = str(tmp_path)
    args.enable_cpu_cache = True
    args.data_type = "bf16"
    with pytest.raises(RuntimeError, match="startup validation complete"):
        _launch_subprocesses(args)
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(vars(args)))
    cached_functions = (
        get_env_start_args,
        get_added_mtp_kv_layer_num,
        get_llm_data_type,
        get_llm_model_class,
        select_mem_manager_class,
        calcu_cpu_cache_meta,
    )
    for function in cached_functions:
        function.cache_clear()
    try:
        assert select_mem_manager_class() is DeepseekV4MemoryManager
        meta = calcu_cpu_cache_meta()
        assert meta.token_page_size == 2048
        assert meta.page_shape == (meta.head_dim,)
        assert meta.calcu_one_page_size() == meta.head_dim
        assert get_env_start_args().llm_kv_type == "None"
    finally:
        for function in cached_functions:
            function.cache_clear()


@pytest.mark.parametrize(
    "page_size,hash_page_size,cpu_page_size",
    [(1, 256, None), (256, 512, None), (256, 256, 4096), (256, 256, None), (256, 512, 4096)],
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
        assert args.page_size == 256
        assert args.linear_att_hash_page_size == hash_page_size
        assert args.cpu_cache_token_page_size == hash_page_size * 8
        assert args.llm_kv_type == "None"
        assert (args.pd_kv_page_num, args.pd_kv_page_size) == (16, 1024)
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
    if cpu_page_size is not None and cpu_page_size != hash_page_size * 8:
        with pytest.raises(ValueError, match="CPU cache pages must match"):
            _launch_subprocesses(args)
    else:
        with pytest.raises(RuntimeError, match="DSV4 page-size validation passed"):
            _launch_subprocesses(args)


@pytest.mark.parametrize("hash_page_size", [128, 384])
def test_dsv4_rejects_hash_pages_not_aligned_to_physical_kv_pages(monkeypatch, hash_page_size):
    for name in (
        "_set_envs_and_config",
        "auto_set_max_req_total_len",
        "auto_set_fused_shared_experts",
        "set_unique_server_name",
    ):
        monkeypatch.setattr("lightllm.server.api_start." + name, lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.get_model_type", lambda _: "deepseek_v4")
    monkeypatch.setattr("lightllm.server.api_start.is_hybrid_att_model", lambda _: True)
    args = StartArgs(
        model_dir="unused",
        linear_att_hash_page_size=hash_page_size,
        disable_vision=True,
        disable_audio=True,
        disable_shm_warning=True,
    )
    with pytest.raises(ValueError, match="--linear_att_hash_page_size must be divisible by --page_size"):
        _launch_subprocesses(args)


def test_pd_cli_and_dataclass_keep_main_defaults():
    import argparse
    from lightllm.server.api_cli import add_cli_args

    parsed = add_cli_args(argparse.ArgumentParser()).parse_args([])
    args = StartArgs()
    assert (parsed.pd_kv_page_num, parsed.pd_kv_page_size) == (16, 1024)
    assert (args.pd_kv_page_num, args.pd_kv_page_size) == (16, 1024)


def test_pd_master_does_not_apply_model_local_cpu_cache_restrictions(monkeypatch):
    from lightllm.server.api_start import pd_master_start

    monkeypatch.setattr("lightllm.server.api_start._set_envs_and_config", lambda args: None)
    monkeypatch.setattr("lightllm.server.api_start.set_unique_server_name", lambda args: None)

    def validation_finished(args):
        raise RuntimeError("PD master uses no local CPU cache")

    monkeypatch.setattr("lightllm.server.api_start.auto_set_max_req_total_len", validation_finished)
    pd_master_args = StartArgs(run_mode="pd_master", model_dir="unused", enable_cpu_cache=True)
    with pytest.raises(RuntimeError, match="PD master uses no local CPU cache"):
        pd_master_start(pd_master_args)


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
