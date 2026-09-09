import json
from types import SimpleNamespace

import pytest
import torch

import lightllm.common.kv_cache_mem_manager.fp8_static_per_head_quant_mem_manager as fp8_module
from lightllm.common.kv_cache_mem_manager.fp8_static_per_head_quant_mem_manager import FP8StaticPerHeadQuantMemManager


def _joint_config():
    # Two target rows followed by one draft row, with four global KV heads.
    rows = []
    for base in (1, 21, 41):
        rows.append([base, base + 1, base + 2, base + 3, base + 10, base + 11, base + 12, base + 13])
    return {
        "version": "1.0",
        "architectures": "TestModel",
        "quant_type": "per_head",
        "qmin": torch.finfo(torch.float8_e4m3fn).min,
        "qmax": torch.finfo(torch.float8_e4m3fn).max,
        "num_layers": 3,
        "num_target_layers": 2,
        "num_draft_layers": 1,
        "num_head": 4,
        "scales_shape": [3, 8],
        "scales": rows,
    }


def _validator(layer_num, draft_layers):
    manager = object.__new__(FP8StaticPerHeadQuantMemManager)
    manager.layer_num = layer_num
    return manager


def test_joint_config_layout_validation_and_target_only_compatibility(monkeypatch):
    cfg = _joint_config()
    manager = _validator(layer_num=3, draft_layers=1)
    monkeypatch.setattr(fp8_module, "get_added_mtp_kv_layer_num", lambda: 1)
    manager._validate_config_layout_and_scales(cfg)

    # The same config is valid on a target-only model; its loader consumes the
    # leading two target rows after this validation.
    target_only = _validator(layer_num=2, draft_layers=0)
    monkeypatch.setattr(fp8_module, "get_added_mtp_kv_layer_num", lambda: 0)
    target_only._validate_config_layout_and_scales(cfg)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cfg: cfg.update(num_target_layers=1), "target\\+draft"),
        (lambda cfg: cfg.update(num_draft_layers=2), "target\\+draft"),
        (lambda cfg: cfg.update(num_layers=3.0), "invalid FP8 KV calibration"),
        (lambda cfg: cfg.update(scales_shape=[3, 7]), "scales_shape"),
        (lambda cfg: cfg["scales"][0].__setitem__(0, float("nan")), "finite"),
        (lambda cfg: cfg["scales"][0].__setitem__(0, 0), "positive"),
    ],
)
def test_joint_config_rejects_invalid_metadata_or_scales(monkeypatch, mutation, message):
    cfg = _joint_config()
    mutation(cfg)
    manager = _validator(layer_num=3, draft_layers=1)
    monkeypatch.setattr(fp8_module, "get_added_mtp_kv_layer_num", lambda: 1)
    with pytest.raises(ValueError, match=message):
        manager._validate_config_layout_and_scales(cfg)


def test_legacy_exact_layout_remains_compatible(monkeypatch):
    cfg = _joint_config()
    cfg.pop("num_target_layers")
    cfg.pop("num_draft_layers")
    manager = _validator(layer_num=3, draft_layers=1)
    monkeypatch.setattr(fp8_module, "get_added_mtp_kv_layer_num", lambda: 1)
    manager._validate_config_layout_and_scales(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FP8 KV scale loader")
@pytest.mark.parametrize(
    ("tp_world_size", "head_num", "rank", "draft_layers", "expected"),
    [
        (2, 2, 1, 1, [[3, 4, 13, 14], [23, 24, 33, 34], [43, 44, 53, 54]]),
        (4, 1, 3, 0, [[4, 14], [24, 34]]),
    ],
)
def test_joint_config_loads_for_target_draft_and_target_only_tp_layouts(
    monkeypatch, tmp_path, tp_world_size, head_num, rank, draft_layers, expected
):
    path = tmp_path / "joint.json"
    path.write_text(json.dumps(_joint_config()))
    args = SimpleNamespace(kv_quant_calibration_config_path=str(path), model_dir="test-model")
    monkeypatch.setattr(fp8_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(fp8_module, "get_model_architectures", lambda _: "TestModel")
    monkeypatch.setattr(fp8_module, "get_dp_world_size", lambda: tp_world_size)
    monkeypatch.setattr(fp8_module, "get_current_rank_in_dp", lambda: rank)
    monkeypatch.setattr(fp8_module, "get_added_mtp_kv_layer_num", lambda: draft_layers)
    monkeypatch.setenv("LIGHTLLM_CURRENT_RANK_IN_NODE", str(rank))
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", f"joint-config-test-{tp_world_size}-{rank}")

    layer_num = 3 if draft_layers else 2
    manager = FP8StaticPerHeadQuantMemManager(
        size=1, dtype=torch.bfloat16, head_num=head_num, head_dim=1, layer_num=layer_num
    )
    assert manager.scales.cpu().tolist() == expected
