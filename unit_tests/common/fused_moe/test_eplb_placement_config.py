import json

import pytest

from lightllm.server.router.model_infer.mode_backend.eplb.placement import (
    load_layer_placement,
    save_placement_config,
)
from lightllm.server.router.model_infer.mode_backend.eplb.placement import config as config_module


@pytest.mark.parametrize("run_mode", ["prefill", "decode"])
@pytest.mark.parametrize(
    "layer_placement",
    [
        [[0, 1, 2], [2, 3, 0]],  # 规范主槽。
        [[0, 1, 3], [2, 3, 1]],  # 只修改冗余槽，仍然合法。
        [[1, 0, 2], [2, 3, 0]],  # 主专家集合相同，但槽序不规范。
        [[0, 2, 1], [2, 3, 0]],  # 主专家移入冗余槽。
        [[0, 1, 2], [0, 1, 3]],  # 全局覆盖完整，但 rank 1 缺少主专家 2。
        [[2, 3, 0], [0, 1, 2]],  # 主专家跨 rank 重排。
    ],
)
def test_load_placement_checks_canonical_primary_slots_only_for_decode(
    tmp_path, monkeypatch, run_mode, layer_placement
):
    config_path = str(tmp_path / "placement.json")
    assert save_placement_config(
        config_path=config_path,
        layer_indexes=[3],
        placement=[layer_placement],
        num_logical_experts=4,
        world_size=2,
        num_redundant_experts_per_rank=1,
    )
    warnings = []
    monkeypatch.setattr(config_module.logger, "warning", lambda *args: warnings.append(args))
    loaded = load_layer_placement(
        config_path=config_path,
        layer_index=3,
        num_logical_experts=4,
        world_size=2,
        num_redundant_experts_per_rank=1,
        run_mode=run_mode,
    )

    canonical = layer_placement[0][:2] == [0, 1] and layer_placement[1][:2] == [2, 3]
    if run_mode == "decode" and not canonical:
        assert loaded is None
        assert len(warnings) == 1
        assert "canonical primary slots" in str(warnings[0][3])
        assert "using the default initial placement" in warnings[0][0]
    else:
        assert loaded == layer_placement
        assert loaded is not layer_placement
        assert not warnings


def test_placement_config_round_trip(tmp_path):
    config_path = tmp_path / "nested" / "eplb-placement.json"
    placement = [
        [[0, 1, 2], [2, 3, 0]],
        [[1, 0, 3], [2, 3, 1]],
    ]

    assert save_placement_config(
        str(config_path),
        layer_indexes=[3, 7],
        placement=placement,
        num_logical_experts=4,
        world_size=2,
        num_redundant_experts_per_rank=1,
    )
    assert (
        load_layer_placement(
            str(config_path),
            layer_index=7,
            num_logical_experts=4,
            world_size=2,
            num_redundant_experts_per_rank=1,
        )
        == placement[1]
    )

    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_config == {
        "version": 1,
        "num_logical_experts": 4,
        "world_size": 2,
        "num_redundant_experts_per_rank": 1,
        "layers": {"3": placement[0], "7": placement[1]},
    }
    assert not (config_path.parent / f"{config_path.name}.lock").exists()


def test_placement_config_lock_prevents_concurrent_write(tmp_path, monkeypatch):
    config_path = tmp_path / "eplb-placement.json"
    lock_path = tmp_path / "eplb-placement.json.lock"
    config_path.write_text("original", encoding="utf-8")
    lock_path.write_text("another-process", encoding="utf-8")
    warnings = []
    monkeypatch.setattr(config_module.logger, "warning", lambda *args: warnings.append(args))

    assert not save_placement_config(
        str(config_path),
        layer_indexes=[3],
        placement=[[[0, 1, 2], [2, 3, 0]]],
        num_logical_experts=4,
        world_size=2,
        num_redundant_experts_per_rank=1,
    )
    assert config_path.read_text(encoding="utf-8") == "original"
    assert lock_path.read_text(encoding="utf-8") == "another-process"
    assert len(warnings) == 1


def test_missing_placement_config_warns_and_falls_back(tmp_path, monkeypatch):
    config_path = tmp_path / "missing.json"
    warnings = []
    config_module._read_config.cache_clear()
    monkeypatch.setattr(config_module.logger, "warning", lambda *args: warnings.append(args))

    assert (
        load_layer_placement(
            str(config_path),
            layer_index=3,
            num_logical_experts=4,
            world_size=2,
            num_redundant_experts_per_rank=1,
        )
        is None
    )
    assert len(warnings) == 1
    assert "using the default initial placement" in warnings[0][0]


def test_malformed_json_warns_and_falls_back(tmp_path, monkeypatch):
    config_path = tmp_path / "malformed.json"
    config_path.write_text("{not-json", encoding="utf-8")
    warnings = []
    config_module._read_config.cache_clear()
    monkeypatch.setattr(config_module.logger, "warning", lambda *args: warnings.append(args))

    assert (
        load_layer_placement(
            str(config_path),
            layer_index=3,
            num_logical_experts=4,
            world_size=2,
            num_redundant_experts_per_rank=1,
        )
        is None
    )
    assert len(warnings) == 1
    assert "using the default initial placement" in warnings[0][0]


def test_invalid_placement_config_warns_and_falls_back(tmp_path, monkeypatch):
    config_path = tmp_path / "invalid.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "num_logical_experts": 4,
                "world_size": 2,
                "num_redundant_experts_per_rank": 1,
                "layers": {"3": [[0, 1, 1], [2, 3, 0]]},
            }
        ),
        encoding="utf-8",
    )
    warnings = []
    config_module._read_config.cache_clear()
    monkeypatch.setattr(config_module.logger, "warning", lambda *args: warnings.append(args))

    assert (
        load_layer_placement(
            str(config_path),
            layer_index=3,
            num_logical_experts=4,
            world_size=2,
            num_redundant_experts_per_rank=1,
        )
        is None
    )
    assert len(warnings) == 1
    assert "contains duplicate expert IDs" in str(warnings[0][3])


def test_missing_layer_warns_and_falls_back(tmp_path, monkeypatch):
    config_path = tmp_path / "missing-layer.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "num_logical_experts": 4,
                "world_size": 2,
                "num_redundant_experts_per_rank": 1,
                "layers": {},
            }
        ),
        encoding="utf-8",
    )
    warnings = []
    config_module._read_config.cache_clear()
    monkeypatch.setattr(config_module.logger, "warning", lambda *args: warnings.append(args))

    assert (
        load_layer_placement(
            str(config_path),
            layer_index=3,
            num_logical_experts=4,
            world_size=2,
            num_redundant_experts_per_rank=1,
        )
        is None
    )
    assert len(warnings) == 1
    assert "is missing" in str(warnings[0][3])


def test_topology_mismatch_warns_and_falls_back(tmp_path, monkeypatch):
    config_path = tmp_path / "mismatch.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "num_logical_experts": 8,
                "world_size": 2,
                "num_redundant_experts_per_rank": 1,
                "layers": {},
            }
        ),
        encoding="utf-8",
    )
    warnings = []
    config_module._read_config.cache_clear()
    monkeypatch.setattr(config_module.logger, "warning", lambda *args: warnings.append(args))

    assert (
        load_layer_placement(
            str(config_path),
            layer_index=3,
            num_logical_experts=4,
            world_size=2,
            num_redundant_experts_per_rank=1,
        )
        is None
    )
    assert len(warnings) == 1
    assert "does not match the current deployment" in warnings[0][0]
