"""加权 rank 优先级：本卡固定，远端按节点局部性和覆盖数评分。

与 hash_targets 路由独立，不使用临时目标集合或 prefill 副本候选表。
采用全 rank 矩阵归约，直接为各专家选择评分最高的可用副本。
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from .decode_eplb_route_statistics import _record_decode_route_cooccurrence


@triton.jit
def _eplb_rank_weighted_matrix_kernel(
    logical_topk_ids_ptr,
    physical_topk_ids_ptr,
    logical_to_rank_physical_map_ptr,
    decode_route_counter_ptr,
    decode_route_recording_enabled_ptr,
    num_tokens,
    current_rank,
    decode_route_counter_row_stride,
    WORLD_SIZE: tl.constexpr,
    NODE_WORLD_SIZE: tl.constexpr,
    NORMALIZE_REPLICA_COVERAGE: tl.constexpr,
    UPDATE_ROUTE_COUNTER: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_TOP_K: tl.constexpr,
    BLOCK_RANKS: tl.constexpr,
):
    # 1. 读取 logical ID 与本卡副本；只有本卡缺少副本的专家参与远端评分。
    token_offsets = tl.program_id(axis=0) * BLOCK_TOKENS + tl.arange(start=0, end=BLOCK_TOKENS)  # [BT]：token 行号。
    topk_slots = tl.arange(start=0, end=BLOCK_TOP_K)  # [BK]：补齐后的 top-k 列号。
    rank_offsets = tl.arange(start=0, end=BLOCK_RANKS)  # [BR]：补齐后的目标 rank。
    topk_id_offsets = token_offsets[:, None] * TOP_K + topk_slots[None, :]  # [BT, BK]：输入/输出的展平位置。
    valid_mask = (token_offsets[:, None] < num_tokens) & (topk_slots[None, :] < TOP_K)  # [BT, BK]：有效 token/expert。
    # [BT, BK]：逻辑专家 ID；补齐位置使用 -1，不对应任何真实专家。
    logical_expert_ids = tl.load(
        pointer=logical_topk_ids_ptr + topk_id_offsets,
        mask=valid_mask,
        other=-1,
    )
    rank_map_row_offsets = logical_expert_ids * WORLD_SIZE  # [BT, BK]：专家在 [E, W] 副本表中的行起点。
    # [BT, BK]：本卡物理槽；无副本和补齐位置均为 -1。
    local_physical_ids = tl.load(
        pointer=logical_to_rank_physical_map_ptr + rank_map_row_offsets + current_rank,
        mask=valid_mask,
        other=-1,
    )
    remote_mask = valid_mask & (local_physical_ids < 0)  # [BT, BK]：需要选择远端副本的专家。

    # 2. 查询全部 rank 的副本，沿 top-k 维累加各专家的覆盖贡献。
    candidate_mask = remote_mask[:, :, None] & (rank_offsets[None, None, :] < WORLD_SIZE)  # [BT, BK, BR]：有效远端候选位置。
    candidate_physical_ids = tl.load(  # [BT, BK, BR]：各专家在各 rank 的槽，无副本/无效位置为 -1。
        pointer=logical_to_rank_physical_map_ptr + rank_map_row_offsets[:, :, None] + rank_offsets[None, None, :],
        mask=candidate_mask,
        other=-1,
    )
    replica_available_mask = candidate_physical_ids >= 0  # [BT, BK, BR]：该 rank 是否能执行该远端专家。
    if NORMALIZE_REPLICA_COVERAGE:
        replica_counts = tl.sum(input=replica_available_mask.to(tl.int32), axis=2)  # [BT, BK]：各远端专家的全局副本 rank 数。
        # 本卡命中和补齐位置的计数为 0，夹到 1 仅避免除零；可用性 mask 保证它们不贡献分数。
        expert_weights = 1024 // tl.maximum(x=replica_counts, y=1)  # [BT, BK]：整数倒数权重，三副本为 341。
        coverage_scores = tl.sum(  # [BT, BR]：各 rank 的稀缺性加权覆盖分；0 是评分贡献的中性值，不是物理槽号。
            input=tl.where(condition=replica_available_mask, x=expert_weights[:, :, None], y=0),
            axis=1,
        )
    else:
        coverage_scores = tl.sum(input=replica_available_mask.to(tl.int32), axis=1)  # [BT, BR]：原始覆盖数，每个远端专家贡献 1。

    # 3. 同 token 共用评分：同节点加 10000000，其次比较覆盖分；节点加分始终高于最大覆盖分。
    same_node_mask = (
        rank_offsets[None, :] // NODE_WORLD_SIZE == current_rank // NODE_WORLD_SIZE
    )  # [1, BR]：目标 rank 是否在当前节点。
    node_scores = same_node_mask.to(tl.int32) * 10000000  # [1, BR]：固定节点优先级。
    rank_scores = node_scores + coverage_scores  # [BT, BR]：节点优先，其次覆盖分，越大越优先。

    # 4. 各专家仅在有副本的 rank 中选最大分；-1 屏蔽无副本位置，合法分数均非负。
    candidate_scores = tl.where(  # [BT, BK, BR]：按副本可用性屏蔽后的分数。
        condition=replica_available_mask,
        x=rank_scores[:, None, :],
        y=-1,
    )
    # 同分时允许归约选择任一最高分 rank，不约束较小 rank；仍只取一个副本。
    best_rank_indices = tl.argmax(  # [BT, BK, 1]：每个专家选中的 rank 列号。
        input=candidate_scores,
        axis=2,
        tie_break_left=False,
        keep_dims=True,
    )
    selected_physical_ids = tl.gather(  # [BT, BK, 1]：从已加载的副本矩阵取出单个槽，无需再次查表。
        src=candidate_physical_ids,
        index=best_rank_indices,
        axis=2,
    )
    # rank 维已经只剩一个槽，沿该维求和只移除单元素维度，不合并多个副本。
    remote_physical_ids = tl.sum(input=selected_physical_ids, axis=2)  # [BT, BK]：每个专家唯一选中的远端槽。
    # [BT, BK]：本卡优先的最终 physical ID，top-k 槽顺序保持不变。
    physical_topk_ids = tl.where(
        condition=local_physical_ids >= 0,
        x=local_physical_ids,
        y=remote_physical_ids,
    )
    tl.store(pointer=physical_topk_ids_ptr + topk_id_offsets, value=physical_topk_ids, mask=valid_mask)

    # 5. 按原始 logical ID 记录共现，不让物理副本选择改变统计含义。
    if UPDATE_ROUTE_COUNTER:
        _record_decode_route_cooccurrence(
            logical_topk_ids_ptr=logical_topk_ids_ptr,
            topk_id_offsets=topk_id_offsets,
            row_expert_ids=logical_expert_ids,
            valid_mask=valid_mask,
            num_topk_ids=num_tokens * TOP_K,
            decode_route_counter_ptr=decode_route_counter_ptr,
            decode_route_counter_row_stride=decode_route_counter_row_stride,
            decode_route_recording_enabled_ptr=decode_route_recording_enabled_ptr,
            TOP_K=TOP_K,
        )


@torch.no_grad()
def eplb_rank_weighted_topk_ids(
    logical_topk_ids: torch.Tensor,
    logical_to_rank_physical_map: torch.Tensor,
    current_rank: int,
    node_world_size: int,
    route_counter: Optional[torch.Tensor] = None,
    update_route_counter: bool = False,
    route_recording_enabled: Optional[torch.Tensor] = None,
    normalize_replica_coverage: bool = True,
) -> torch.Tensor:
    """以节点优先、覆盖数次优的共同分数路由；分块在 Graph 捕获时固定。

    logical_topk_ids [T,K]，要求 T/K 均大于 0；logical_to_rank_physical_map [E,W]。本卡专家始终本地执行。
    评分为 same_node * 10000000 + coverage_score，不包含可调系数或随机/哈希扰动。
    默认 normalize_replica_coverage=True，每个专家贡献 1024 // 全局副本 rank 数；False 时贡献 1。
    归一化只做整数向下取整，不再次缩小累计分数；副本数超过 1024 时贡献为 0。
    本卡命中不参与覆盖评分，有本节点副本时仅在本节点内选，否则全局选覆盖分最高的 rank。
    同分时允许选择任一最高分可用 rank，具体选择依赖归约布局，不保证负载均匀。
    覆盖数促进共址，但不是接收端负载惩罚，也不保证最小通信目标数。
    所有专家必须至少有一个副本，每 rank 内同一 logical ID 的槽唯一。
    执行布局由内部按 token 数、top-k、rank 数及统计配置确定，Graph 捕获后固定，不提供外部调优参数。
    """
    assert logical_topk_ids.ndim == 2 and logical_topk_ids.is_contiguous()
    assert logical_topk_ids.dtype in (torch.int32, torch.int64)
    assert logical_topk_ids.numel() > 0, "logical_topk_ids must be non-empty"
    assert logical_to_rank_physical_map.ndim == 2 and logical_to_rank_physical_map.is_contiguous()
    assert logical_to_rank_physical_map.dtype is torch.int32
    assert logical_topk_ids.device == logical_to_rank_physical_map.device
    world_size = logical_to_rank_physical_map.shape[1]
    assert 0 <= current_rank < world_size and 0 < node_world_size <= world_size and world_size % node_world_size == 0
    assert isinstance(normalize_replica_coverage, bool)
    # 单专家贡献最多 1 或 1024；最大覆盖分必须小于节点加分，也保证总分不溢出 int32。
    assert logical_topk_ids.shape[1] * (1024 if normalize_replica_coverage else 1) < 10000000
    block_ranks = triton.next_power_of_2(n=world_size)
    block_top_k = triton.next_power_of_2(n=logical_topk_ids.shape[1])
    # 默认每 token 一个 program：小 batch 或较宽矩阵不宜增大分块，避免归约和寄存器开销。
    block_tokens = 1
    num_warps = 1 if block_top_k * block_ranks <= 128 else 4
    # H200 上 K=8、W=8/16/32 的多轮 Graph 评测：大 batch 合并两行可减少 program 数。
    # 较宽矩阵需要更多 token 才能抵消合并开销；其他 K/W 保留默认配置，不外推未验证的布局。
    if logical_topk_ids.shape[1] == 8 and world_size in (8, 16, 32):
        multi_token_threshold = {8: 512, 16: 1536, 32: 2048}[world_size]
        block_tokens = 2 if logical_topk_ids.shape[0] >= multi_token_threshold else 1
        # 大 batch 已有足够 program 提供并行度，1 warp 避免宽矩阵下额外的跨 warp 归约。
        if logical_topk_ids.shape[0] >= 1024:
            num_warps = 1
        # W=16 的超大 batch 在纯映射路径合并四行更快；含统计的 kernel 保留两行，避免原子操作开销。
        # 不读取设备统计开关来选择布局，Graph 捕获后的开关变化不会触发布局切换。
        if world_size == 16 and logical_topk_ids.shape[0] >= 8192 and not update_route_counter:
            block_tokens = 4
    if update_route_counter:
        experts = logical_to_rank_physical_map.shape[0]
        assert route_counter is not None and route_counter.shape == (experts, experts)
        assert route_counter.dtype is torch.int64 and route_counter.is_contiguous()
        assert route_recording_enabled is not None and route_recording_enabled.shape == (1,)
        assert route_recording_enabled.dtype is torch.int32 and route_recording_enabled.is_contiguous()
        assert route_counter.device == logical_topk_ids.device == route_recording_enabled.device
        counter, enabled = route_counter, route_recording_enabled
    else:
        # 不记录统计时仅传占位指针；constexpr 裁剪后不会访问统计表或开关。
        counter, enabled = logical_to_rank_physical_map, logical_to_rank_physical_map
    output = torch.empty_like(input=logical_topk_ids)
    _eplb_rank_weighted_matrix_kernel[(triton.cdiv(x=logical_topk_ids.shape[0], y=block_tokens),)](
        logical_topk_ids_ptr=logical_topk_ids,
        physical_topk_ids_ptr=output,
        logical_to_rank_physical_map_ptr=logical_to_rank_physical_map,
        decode_route_counter_ptr=counter,
        decode_route_recording_enabled_ptr=enabled,
        num_tokens=logical_topk_ids.shape[0],
        current_rank=current_rank,
        decode_route_counter_row_stride=counter.stride(dim=0),
        WORLD_SIZE=world_size,
        NODE_WORLD_SIZE=node_world_size,
        NORMALIZE_REPLICA_COVERAGE=normalize_replica_coverage,
        UPDATE_ROUTE_COUNTER=update_route_counter,
        TOP_K=logical_topk_ids.shape[1],
        BLOCK_TOKENS=block_tokens,
        BLOCK_TOP_K=block_top_k,
        BLOCK_RANKS=block_ranks,
        num_warps=num_warps,
        num_stages=1,
    )
    return output
