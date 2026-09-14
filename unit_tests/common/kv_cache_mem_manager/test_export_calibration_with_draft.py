import json
from types import SimpleNamespace

import pytest
import torch

import lightllm.common.kv_cache_mem_manager.export_calibration_mem_manager as calibration_module
from lightllm.common.kv_cache_mem_manager.export_calibration_mem_manager import ExportCalibrationMemoryManager


def _manager(layer_num=3, head_num=2):
    manager = object.__new__(ExportCalibrationMemoryManager)
    manager.layer_num = layer_num
    manager.head_num = head_num
    manager.total_head_num = head_num
    manager.qmax = torch.finfo(torch.float8_e4m3fn).max
    manager.qmin = torch.finfo(torch.float8_e4m3fn).min
    manager.calibration_counts = [0] * layer_num
    manager._calibration_finalized = False
    manager.scales = None
    manager.abs_max = torch.zeros((layer_num, 2 * head_num), dtype=torch.float32)
    return manager


def _kv(value):
    # K heads have maxima value and 2*value; V heads have 3*value and 4*value.
    return torch.tensor([[[value], [2 * value], [3 * value], [4 * value]]], dtype=torch.bfloat16)


def test_joint_export_waits_for_each_target_and_draft_layer(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(calibration_module, "get_model_init_status", lambda: True)
    monkeypatch.setattr(calibration_module, "get_kv_quant_calibration_warmup_count", lambda: 1)
    monkeypatch.setattr(calibration_module, "get_kv_quant_calibration_inference_count", lambda: 2)
    monkeypatch.setattr(calibration_module, "get_added_mtp_kv_layer_num", lambda: 1)
    monkeypatch.setattr(calibration_module, "get_global_rank", lambda: 0)
    monkeypatch.setattr(calibration_module, "get_model_architectures", lambda _: "TestModel")
    monkeypatch.setattr(
        calibration_module,
        "get_env_start_args",
        lambda: SimpleNamespace(llm_prefill_att_backend=["fa3"], model_dir="test-model"),
    )
    manager = _manager()

    # The final (draft) layer finishes first. It must not write a partial file
    # while target rows still have no post-warmup samples.
    for value in (1, 2, 7):
        manager.update_calibration_data(_kv(value), 2)
    assert not (tmp_path / "kv_cache_calib_per_head_with_draft.json").exists()
    assert manager.calibration_counts == [0, 0, 3]

    for value in (1, 3, 5):
        manager.update_calibration_data(_kv(value), 0)
    assert not (tmp_path / "kv_cache_calib_per_head_with_draft.json").exists()

    for value in (1, 4, 6):
        manager.update_calibration_data(_kv(value), 1)

    path = tmp_path / "kv_cache_calib_per_head_with_draft.json"
    cfg = json.loads(path.read_text())
    assert manager._calibration_finalized
    assert manager.calibration_counts == [3, 3, 3]
    assert cfg["num_layers"] == 3
    assert cfg["num_target_layers"] == 2
    assert cfg["num_draft_layers"] == 1
    assert cfg["scales_shape"] == [3, 4]
    torch.testing.assert_close(
        torch.tensor(cfg["scales"]),
        torch.tensor([[5, 10, 15, 20], [6, 12, 18, 24], [7, 14, 21, 28]], dtype=torch.float32) / manager.qmax,
    )

    # DSpark may update draft rows twice in a logical step. Extra writes after
    # their individual quota do not collect more data or re-export the file.
    manager.update_calibration_data(_kv(99), 2)
    assert manager.calibration_counts == [3, 3, 3]
    assert json.loads(path.read_text()) == cfg


def test_empty_kv_does_not_consume_a_layer_sample(monkeypatch):
    monkeypatch.setattr(calibration_module, "get_model_init_status", lambda: True)
    monkeypatch.setattr(calibration_module, "get_kv_quant_calibration_warmup_count", lambda: 0)
    monkeypatch.setattr(calibration_module, "get_kv_quant_calibration_inference_count", lambda: 1)
    monkeypatch.setattr(
        calibration_module,
        "get_env_start_args",
        lambda: SimpleNamespace(llm_prefill_att_backend=["fa3"], model_dir="test-model"),
    )
    manager = _manager(layer_num=1)
    manager.update_calibration_data(torch.empty((0, 4, 1), dtype=torch.bfloat16), 0)
    assert manager.calibration_counts == [0]


def test_nonfinite_kv_data_fails_calibration(monkeypatch):
    monkeypatch.setattr(calibration_module, "get_model_init_status", lambda: True)
    monkeypatch.setattr(calibration_module, "get_kv_quant_calibration_warmup_count", lambda: 0)
    monkeypatch.setattr(calibration_module, "get_kv_quant_calibration_inference_count", lambda: 1)
    monkeypatch.setattr(
        calibration_module,
        "get_env_start_args",
        lambda: SimpleNamespace(llm_prefill_att_backend=["fa3"], model_dir="test-model"),
    )
    manager = _manager(layer_num=1)
    with pytest.raises(ValueError, match="non-finite"):
        manager.update_calibration_data(torch.full((1, 4, 1), float("nan"), dtype=torch.bfloat16), 0)
