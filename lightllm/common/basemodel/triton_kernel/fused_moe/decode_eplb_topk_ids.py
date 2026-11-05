"""Decode EPLB 专家映射的统一入口，默认按 token 规模选择最小映射或加权路由。"""

from typing import Optional

import torch

from lightllm.utils.envs_utils import get_eplb_decode_routing_mode
from .decode_eplb_topk_ids_hash_targets import eplb_hash_targets_topk_ids
from .decode_eplb_topk_ids_minimal import eplb_minimal_topk_ids
from .decode_eplb_topk_ids_rank_weighted import eplb_rank_weighted_topk_ids


__all__ = ["eplb_repair_decode_topk_ids"]

# TODO：512 是待调优的启发式阈值，尚未通过两条路由的性能交叉点确定。
# 后续在相同布局和 token 输入下，对比最小映射与加权路由的稳态 decode 完整耗时、
# 通信/GEMM 耗时及最慢 rank 负载，结合 E、top-k、冗余数和节点拓扑调优。
# 注意 T 包含 Graph 补齐行，T*W 不一定等于实际全局 token 数；可按 Graph 档位静态选路。
_MINIMAL_ROUTING_TOKEN_THRESHOLD = 512


def eplb_repair_decode_topk_ids(
    logical_topk_ids: torch.Tensor,  # [T, K]：输入 logical ID，int32/int64。
    logical_to_physical_map: torch.Tensor,  # [E, 3 + W*S]：复用 prefill 表的 global_count/node_count 和全部 physical ID。
    logical_to_rank_physical_map: torch.Tensor,  # [E, W]：各专家在各 rank 的 physical ID，缺少副本时为 -1。
    minimal_logical_to_physical_map: torch.Tensor,  # [E]：每个逻辑专家固定使用的唯一物理槽。
    current_rank: int,  # 标量：当前源 rank。
    num_physical_experts_per_rank: int,  # 标量：每卡物理槽数。
    route_counter: Optional[torch.Tensor] = None,  # [E, E] 或 None：logical 专家共现计数，int64。
    update_route_counter: bool = False,  # 标量：是否保留统计分支。
    route_recording_enabled: Optional[torch.Tensor] = None,  # [1] 或 None：运行时统计开关，int32。
    node_world_size: int = 1,  # 标量：每节点 rank 数，默认每节点一个 rank。
) -> torch.Tensor:
    """根据 LIGHTLLM_EPLB_DECODE_ROUTING_MODE 选择 decode 专家映射算法。

    default_targets（默认）：T*W < 512 时直接使用最小映射；否则本卡优先，远端按节点和倒数加权覆盖分选副本。
    hash_targets：本卡优先，token 级伪随机初选优先本节点，再在临时目标 rank 中共址。
    两种模式均保持 logical ID 与 top-k 顺序不变，返回同 shape/dtype 的 physical ID。
    小批量路径不做本卡副本优先，所有源 rank 共用同一最小映射，控制活跃物理专家规模。
    T 是输入 tensor 行数，Graph 捕获时包含补齐行；T*W 不是运行时跨卡 token 数求和。
    T*W == 512 使用原加权矩阵路由，两条路径均不保证通信目标数或运行时负载最优。
    node_world_size 用于 hash_targets 的初选范围，以及 default_targets 的节点优先级。

    环境变量首次读取后缓存，必须在各 worker 启动、CUDA Graph 捕获前统一设置。
    Graph 重放不会重新进入 Python 选择算法；修改模式需要重启进程并重新捕获。
    """
    # 这里只转发参数，不额外启动 GPU kernel；加权路由不需要 prefill 候选表或物理槽数。
    if get_eplb_decode_routing_mode() == "hash_targets":
        return eplb_hash_targets_topk_ids(
            logical_topk_ids=logical_topk_ids,
            logical_to_physical_map=logical_to_physical_map,
            logical_to_rank_physical_map=logical_to_rank_physical_map,
            current_rank=current_rank,
            num_physical_experts_per_rank=num_physical_experts_per_rank,
            route_counter=route_counter,
            update_route_counter=update_route_counter,
            route_recording_enabled=route_recording_enabled,
            node_world_size=node_world_size,
        )
    assert logical_to_rank_physical_map.ndim == 2
    world_size = logical_to_rank_physical_map.shape[1]
    assert world_size > 0
    if logical_topk_ids.shape[0] * world_size < _MINIMAL_ROUTING_TOKEN_THRESHOLD:
        assert minimal_logical_to_physical_map.shape == (logical_to_rank_physical_map.shape[0],)
        return eplb_minimal_topk_ids(
            logical_topk_ids=logical_topk_ids,
            minimal_logical_to_physical_map=minimal_logical_to_physical_map,
            route_counter=route_counter,
            update_route_counter=update_route_counter,
            route_recording_enabled=route_recording_enabled,
        )
    return eplb_rank_weighted_topk_ids(
        logical_topk_ids=logical_topk_ids,
        logical_to_rank_physical_map=logical_to_rank_physical_map,
        current_rank=current_rank,
        node_world_size=node_world_size,
        route_counter=route_counter,
        update_route_counter=update_route_counter,
        route_recording_enabled=route_recording_enabled,
    )
