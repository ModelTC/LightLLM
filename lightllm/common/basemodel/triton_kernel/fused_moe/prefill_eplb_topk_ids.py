"""Prefill EPLB 专家映射的统一入口，按 token 规模选择最小映射或副本哈希。"""

from typing import Optional

import torch

from .prefill_eplb_topk_ids_minimal import eplb_prefill_minimal_topk_ids
from .prefill_eplb_topk_ids_replica_hash import eplb_prefill_replica_hash_topk_ids


__all__ = ["eplb_repair_prefill_topk_ids"]

# TODO：512 是待调优的启发式阈值，需用相同布局和输入对比完整 prefill 的通信/GEMM 耗时，
# 结合专家数、top-k、冗余数和节点拓扑调优，不能直接套用 decode 的性能交叉点。
# T 是本 rank 的输入 token 行数，包含补齐行；T*W 并非实际跨 rank token 数求和。
_MINIMAL_ROUTING_TOKEN_THRESHOLD = 512


def eplb_repair_prefill_topk_ids(
    logical_topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    minimal_logical_to_physical_map: torch.Tensor,
    world_size: int,
    prefill_route_counter: Optional[torch.Tensor],
    prefill_route_sample_index: Optional[torch.Tensor],
    update_prefill_route_counter: bool,
    mode: str,
) -> torch.Tensor:
    """小批量共用唯一物理槽，大批量保留显式指定的 prefill 副本分发策略。

    logical_topk_ids 为 [T, K]；minimal_logical_to_physical_map 为 [E]。
    T*world_size < 512 时直接查最小映射，不优先使用本卡或本节点冗余副本，
    避免相同逻辑专家分散到多个物理槽；否则使用 mode 指定的副本哈希策略。
    两条路径均保持 logical ID 输入不变，并沿用原有逻辑专家环形采样。
    world_size 为 EP rank 数，而非每节点 rank 数；T 不表示请求并发数。
    """
    assert mode in (
        "current_gpu_first",
        "current_node_first",
        "global_first",
    ), f"unsupported EPLB dispatch mode {mode!r}"
    assert logical_topk_ids.ndim == 2
    assert logical_topk_ids.dtype in (torch.int32, torch.int64)
    assert world_size > 0
    if logical_topk_ids.shape[0] * world_size < _MINIMAL_ROUTING_TOKEN_THRESHOLD:
        assert minimal_logical_to_physical_map.shape == (logical_to_physical_map.shape[0],)
        return eplb_prefill_minimal_topk_ids(
            logical_topk_ids=logical_topk_ids,
            minimal_logical_to_physical_map=minimal_logical_to_physical_map,
            prefill_route_counter=prefill_route_counter,
            prefill_route_sample_index=prefill_route_sample_index,
            update_prefill_route_counter=update_prefill_route_counter,
        )
    return eplb_prefill_replica_hash_topk_ids(
        logical_topk_ids=logical_topk_ids,
        logical_to_physical_map=logical_to_physical_map,
        prefill_route_counter=prefill_route_counter,
        prefill_route_sample_index=prefill_route_sample_index,
        update_prefill_route_counter=update_prefill_route_counter,
        mode=mode,
    )
