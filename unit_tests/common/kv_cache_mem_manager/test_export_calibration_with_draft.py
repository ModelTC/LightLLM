from types import SimpleNamespace
import torch

import lightllm.common.kv_cache_mem_manager.export_calibration_mem_manager as calibration_module
from lightllm.common.kv_cache_mem_manager.export_calibration_mem_manager import ExportCalibrationMemoryManager


def _manager(layer_num=3, head_num=2):
    manager = object.__new__(ExportCalibrationMemoryManager)
    manager.layer_num = layer_num
    manager.head_num = head_num
    manager.total_head_num = head_num
    manager.qmax = 448.0
    manager.qmin = -448.0
    manager.calibration_counts = [0] * layer_num
    manager.observed_token_rows = [0] * layer_num
    manager._calibration_active = False
    manager.abs_max = torch.zeros((layer_num, 2 * head_num), dtype=torch.float32)
    return manager


def _kv(value):
    return torch.tensor([[[value], [2 * value], [3 * value], [4 * value]]], dtype=torch.bfloat16)


def test_explicit_lifecycle_collects_every_target_and_draft_layer(monkeypatch):
    monkeypatch.setattr(calibration_module, "get_model_init_status", lambda: True)
    monkeypatch.setattr(calibration_module, "get_global_rank", lambda: 0)
    monkeypatch.setattr(calibration_module, "get_added_mtp_kv_layer_num", lambda: 1)
    monkeypatch.setattr(
        calibration_module,
        "get_env_start_args",
        lambda: SimpleNamespace(llm_prefill_att_backend=["fa3"], model_dir="fake"),
    )
    monkeypatch.setattr(calibration_module, "get_model_architectures", lambda _: "Fake")
    manager = _manager()
    manager.update_calibration_data(_kv(99), 0)
    assert manager.calibration_counts == [0, 0, 0]
    manager.begin_calibration()
    for layer, value in enumerate((5, 6, 7)):
        manager.update_calibration_data(_kv(value), layer)
    snap = manager.snapshot_calibration()
    assert not snap["active"]
    assert snap["counts"] == [1, 1, 1]
    assert snap["observed_token_rows"] == [1, 1, 1]
    assert snap["abs_max"] == [[5, 10, 15, 20], [6, 12, 18, 24], [7, 14, 21, 28]]
    manager.update_calibration_data(_kv(99), 2)
    assert manager.calibration_counts == [1, 1, 1]


def test_empty_kv_does_not_consume_layer_sample(monkeypatch):
    monkeypatch.setattr(calibration_module, "get_model_init_status", lambda: True)
    monkeypatch.setattr(calibration_module, "get_global_rank", lambda: 0)
    monkeypatch.setattr(calibration_module, "get_added_mtp_kv_layer_num", lambda: 1)
    monkeypatch.setattr(
        calibration_module,
        "get_env_start_args",
        lambda: SimpleNamespace(llm_prefill_att_backend=["fa3"], model_dir="fake"),
    )
    monkeypatch.setattr(calibration_module, "get_model_architectures", lambda _: "Fake")
    manager = _manager(layer_num=1)
    manager.begin_calibration()
    manager.update_calibration_data(torch.empty((0, 4, 1), dtype=torch.bfloat16), 0)
    assert manager.calibration_counts == [0]


def test_joint_per_head_export_loads_for_draft_and_target_only(tmp_path, monkeypatch):
    import json
    import lightllm.common.kv_cache_mem_manager.fp8_static_per_head_quant_mem_manager as loader

    cfg = {
        "version": "1.0",
        "architectures": "Fake",
        "quant_type": "per_head",
        "qmin": -448.0,
        "qmax": 448.0,
        "num_layers": 3,
        "num_target_layers": 2,
        "num_draft_layers": 1,
        "num_head": 2,
        "scales_shape": [3, 4],
        "scales": [[1.0] * 4 for _ in range(3)],
    }
    path = tmp_path / "joint.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(loader, "get_model_architectures", lambda _: "Fake")
    monkeypatch.setattr(
        loader,
        "get_env_start_args",
        lambda: SimpleNamespace(kv_quant_calibration_config_path=str(path), model_dir="fake"),
    )
    for runtime_draft, layer_num in ((1, 3), (0, 2)):
        monkeypatch.setattr(loader, "get_added_mtp_kv_layer_num", lambda d=runtime_draft: d)
        obj = object.__new__(loader.FP8StaticPerHeadQuantMemManager)
        obj.qmin, obj.qmax, obj.layer_num = -448.0, 448.0, layer_num
        assert obj._load_and_check_config()["num_target_layers"] == 2
