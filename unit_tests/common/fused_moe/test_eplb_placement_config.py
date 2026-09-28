import json

from lightllm.server.router.model_infer.mode_backend.eplb.placement import (
    load_layer_placement,
    save_placement_config,
)
from lightllm.server.router.model_infer.mode_backend.eplb.placement import config as config_module


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
