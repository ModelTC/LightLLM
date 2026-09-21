"""EPLB expert placement JSON loading and persistence."""

import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional, Sequence

from lightllm.utils.log_utils import init_logger

from .types import ExpertPlacement, LayerPlacement

logger = init_logger(__name__)

EPLB_PLACEMENT_CONFIG_VERSION = 1


def load_layer_placement(
    config_path: str,
    layer_index: int,
    num_logical_experts: int,
    world_size: int,
    num_redundant_experts_per_rank: int,
) -> Optional[LayerPlacement]:
    """读取并校验一个模型层的布局；任何错误都返回 ``None`` 以触发默认流程。"""
    config = _read_config(config_path)
    if config is None:
        return None

    # 阶段 1：构造当前部署期望的元数据，后续各项校验和 warning 都以此为准。
    expected_metadata = {
        "version": EPLB_PLACEMENT_CONFIG_VERSION,
        "num_logical_experts": num_logical_experts,
        "world_size": world_size,
        "num_redundant_experts_per_rank": num_redundant_experts_per_rank,
    }

    # 阶段 2：逐项显式校验配置元数据，便于直接定位具体的不匹配字段。
    version = config.get("version")
    if version != expected_metadata["version"]:
        logger.warning(
            "EPLB placement config %s does not match the current deployment: version=%r, expected %r; "
            "using the default initial placement",
            config_path,
            version,
            expected_metadata["version"],
        )
        return None

    config_num_logical_experts = config.get("num_logical_experts")
    if config_num_logical_experts != expected_metadata["num_logical_experts"]:
        logger.warning(
            "EPLB placement config %s does not match the current deployment: num_logical_experts=%r, "
            "expected %r; using the default initial placement",
            config_path,
            config_num_logical_experts,
            expected_metadata["num_logical_experts"],
        )
        return None

    config_world_size = config.get("world_size")
    if config_world_size != expected_metadata["world_size"]:
        logger.warning(
            "EPLB placement config %s does not match the current deployment: world_size=%r, expected %r; "
            "using the default initial placement",
            config_path,
            config_world_size,
            expected_metadata["world_size"],
        )
        return None

    config_num_redundant_experts_per_rank = config.get("num_redundant_experts_per_rank")
    if config_num_redundant_experts_per_rank != expected_metadata["num_redundant_experts_per_rank"]:
        logger.warning(
            "EPLB placement config %s does not match the current deployment: "
            "num_redundant_experts_per_rank=%r, expected %r; using the default initial placement",
            config_path,
            config_num_redundant_experts_per_rank,
            expected_metadata["num_redundant_experts_per_rank"],
        )
        return None

    # 阶段 3：读取当前模型层的物理槽布局，并校验形状、ID 范围及专家覆盖关系。
    try:
        layers = config.get("layers")
        assert isinstance(layers, dict), "the layers field must be a JSON object"
        layer_placement = layers.get(str(layer_index))
        assert layer_placement is not None, f"layer {layer_index} is missing"
        _validate_layer_placement(
            layer_placement,
            num_logical_experts=num_logical_experts,
            world_size=world_size,
            num_redundant_experts_per_rank=num_redundant_experts_per_rank,
        )
    except Exception as exc:
        logger.warning(
            "Layer %s in EPLB placement config %s is invalid (%s); using the default initial placement",
            layer_index,
            config_path,
            exc,
        )
        return None

    # 校验后复制一份，避免缓存中的原始 JSON 对象被运行态修改。
    return [list(rank_placement) for rank_placement in layer_placement]


def save_placement_config(
    config_path: str,
    layer_indexes: Sequence[int],
    placement: ExpertPlacement,
    num_logical_experts: int,
    world_size: int,
    num_redundant_experts_per_rank: int,
) -> bool:
    """将完整 EPLB 布局原子写入配置路径，失败时仅记录 warning。"""
    if len(layer_indexes) != len(placement) or len(set(layer_indexes)) != len(layer_indexes):
        logger.warning(
            "Failed to save EPLB placement config %s: layer indexes do not match the placement",
            config_path,
        )
        return False

    for layer_index, layer_placement in zip(layer_indexes, placement):
        try:
            _validate_layer_placement(
                layer_placement,
                num_logical_experts=num_logical_experts,
                world_size=world_size,
                num_redundant_experts_per_rank=num_redundant_experts_per_rank,
            )
        except Exception as exc:
            logger.warning(
                "Failed to save EPLB placement config %s: layer %s is invalid (%s)",
                config_path,
                layer_index,
                exc,
            )
            return False

    config = {
        "version": EPLB_PLACEMENT_CONFIG_VERSION,
        "num_logical_experts": num_logical_experts,
        "world_size": world_size,
        "num_redundant_experts_per_rank": num_redundant_experts_per_rank,
        "layers": {
            str(layer_index): [list(rank_placement) for rank_placement in layer_placement]
            for layer_index, layer_placement in zip(layer_indexes, placement)
        },
    }

    absolute_path = os.path.abspath(config_path)
    parent_dir = os.path.dirname(absolute_path)
    lock_path = f"{absolute_path}.lock"
    lock_acquired = False
    try:
        os.makedirs(parent_dir, exist_ok=True)
        # 通过 x 模式原子创建锁文件，避免多个服务进程同时写入同一个布局文件。
        with open(lock_path, "x", encoding="utf-8") as lock_file:
            lock_acquired = True
            lock_file.write(str(os.getpid()))
        with open(absolute_path, "w", encoding="utf-8") as config_file:
            json.dump(config, config_file, ensure_ascii=False, indent=2)
            config_file.write("\n")
        _read_config.cache_clear()
        return True
    except OSError as exc:
        logger.warning("Failed to save EPLB placement config %s: %s", config_path, exc)
        return False
    finally:
        if lock_acquired:
            try:
                os.unlink(lock_path)
            except OSError as exc:
                logger.warning("Failed to remove EPLB placement lock file %s: %s", lock_path, exc)


def _validate_layer_placement(
    layer_placement: Any,
    num_logical_experts: int,
    world_size: int,
    num_redundant_experts_per_rank: int,
) -> None:
    """按顺序断言单层布局满足当前部署的全部约束。"""
    # 阶段 1：校验 rank 维度和每个 rank 应持有的物理槽数量。
    assert isinstance(layer_placement, list), "the layer is not an array"
    assert len(layer_placement) == world_size, f"the number of ranks is {len(layer_placement)}, expected {world_size}"

    num_physical_experts_per_rank = num_logical_experts // world_size + num_redundant_experts_per_rank
    covered_experts = set()
    for rank, rank_placement in enumerate(layer_placement):
        # 阶段 2：依次校验每个 rank 的布局形状和 expert ID。
        assert isinstance(rank_placement, list), f"the placement for rank {rank} is not an array"
        assert (
            len(rank_placement) == num_physical_experts_per_rank
        ), f"rank {rank} has {len(rank_placement)} physical slots, expected {num_physical_experts_per_rank}"
        assert all(
            not isinstance(expert_id, bool) and isinstance(expert_id, int) for expert_id in rank_placement
        ), f"rank {rank} contains a non-integer expert ID"
        assert all(
            0 <= expert_id < num_logical_experts for expert_id in rank_placement
        ), f"rank {rank} contains an out-of-range expert ID"
        assert len(set(rank_placement)) == len(rank_placement), f"rank {rank} contains duplicate expert IDs"
        covered_experts.update(rank_placement)

    # 阶段 3：确认所有 logical expert 至少存在一个物理副本。
    missing_experts = sorted(set(range(num_logical_experts)) - covered_experts)
    assert not missing_experts, f"logical experts are missing: {missing_experts}"


@lru_cache(maxsize=None)
def _read_config(config_path: str) -> Optional[Dict[str, Any]]:
    """读取并缓存配置，避免模型的每个 MoE 层重复解析同一个文件。"""
    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.warning(
            "Failed to read EPLB placement config %s; using the default initial placement: %s",
            config_path,
            exc,
        )
        return None

    if not isinstance(config, dict):
        logger.warning(
            "The root of EPLB placement config %s must be a JSON object; using the default initial placement",
            config_path,
        )
        return None
    return config
