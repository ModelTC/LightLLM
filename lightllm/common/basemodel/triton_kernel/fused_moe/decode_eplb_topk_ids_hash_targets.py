"""Decode 分发：本卡优先、token 级伪随机初选，再按共同 rank 随机优先级共址。"""

from typing import Optional

import torch
import triton
import triton.language as tl

from .decode_eplb_route_statistics import _record_decode_route_cooccurrence


@triton.jit
def _eplb_hash_targets_topk_ids_kernel(
    logical_topk_ids_ptr,
    physical_topk_ids_ptr,
    logical_to_physical_map_ptr,
    logical_to_physical_map_row_stride,
    logical_to_rank_physical_map_ptr,
    logical_to_rank_physical_map_row_stride,
    num_tokens,
    current_rank,
    num_physical_experts_per_rank,
    decode_route_counter_ptr,
    decode_route_counter_row_stride,
    decode_route_recording_enabled_ptr,
    UPDATE_ROUTE_COUNTER: tl.constexpr,
    RANK_BITS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_TOP_K: tl.constexpr,
):
    # [BT] / [BK]：本 program 的 token 行、补齐后的 top-k 列。
    token_offsets = tl.program_id(axis=0) * BLOCK_TOKENS + tl.arange(start=0, end=BLOCK_TOKENS)
    topk_slots = tl.arange(start=0, end=BLOCK_TOP_K)
    topk_id_offsets = token_offsets[:, None] * TOP_K + topk_slots[None, :]
    valid_mask = (token_offsets[:, None] < num_tokens) & (topk_slots[None, :] < TOP_K)
    # [BT, BK]：logical ID 和本卡 physical ID；无效 lane / 缺少副本均为 -1。
    logical_expert_ids = tl.load(pointer=logical_topk_ids_ptr + topk_id_offsets, mask=valid_mask, other=-1)
    rank_map_row_offsets = logical_expert_ids * logical_to_rank_physical_map_row_stride
    local_physical_ids = tl.load(
        pointer=logical_to_rank_physical_map_ptr + rank_map_row_offsets + current_rank, mask=valid_mask, other=-1
    )
    remote_mask = valid_mask & (local_physical_ids < 0)

    # [BT, 1]：以 source rank 为固定种子，按 token 行号生成 Philox 伪随机数。
    # 同 token 的专家共用该随机数；不包含 logical ID 或 top-k 槽序号。
    # 不维护随机状态，固定输入和布局的 Graph 重放可复现；转 uint32 后再取模。
    route_random = tl.randint(seed=current_rank, offset=token_offsets[:, None], n_rounds=10).to(tl.uint32)

    # 复用 prefill 表 [E, 3 + W*S]：第 0/1 列为 global_count/node_count，
    # 第 3 列起的副本按本卡、本节点、其他节点排列。本卡已由 local_physical_ids 优先处理。
    # [BT, BK]：有本节点副本时仅在节点前缀中初选，没有才在全局前缀中初选。
    # 本地及无效 lane 不查询候选；count 的占位值 1 仅用于避免屏蔽 lane 的取模除零。
    candidate_row_offsets = logical_expert_ids * logical_to_physical_map_row_stride
    global_replica_counts = tl.load(
        pointer=logical_to_physical_map_ptr + candidate_row_offsets, mask=remote_mask, other=1
    )
    # 每节点只有一个 rank 时，远端专家的 node_count 自然为 0，同样回退到全局。
    node_replica_counts = tl.load(
        pointer=logical_to_physical_map_ptr + candidate_row_offsets + 1, mask=remote_mask, other=0
    )
    replica_counts = tl.where(condition=node_replica_counts > 0, x=node_replica_counts, y=global_replica_counts)
    candidate_indices = route_random % replica_counts.to(tl.uint32)
    sampled_physical_ids = tl.load(
        pointer=logical_to_physical_map_ptr + candidate_row_offsets + 3 + candidate_indices,
        mask=remote_mask,
        other=-1,
    )
    # [BT, BK]：临时目标先包含真实的本地执行位置，避免本地专家引入远端锚点。
    initial_physical_ids = tl.where(condition=local_physical_ids >= 0, x=local_physical_ids, y=sampled_physical_ids)
    initial_ranks = initial_physical_ids // num_physical_experts_per_rank

    # 标量 seed：低 32 位为 source rank，高 32 位置 1，与初选随机流区分。
    rank_priority_seed = tl.cast(input=current_rank, dtype=tl.uint64) | (1 << 32)
    # [BT, BK]：uint32 offset = (token 行号 << RANK_BITS) | target rank。
    # 低 RANK_BITS 位预留给 rank，高位放 token；位数按 ceil(log2(world_size)) 计算。
    # 非二次幂 world_size 也不会重叠；uint32 溢出按模 2**32 回绕。
    # 不包含 logical ID 或 top-k 槽序号；同 token 对同一 rank 的评分始终相同。
    rank_random_offsets = (token_offsets[:, None].to(tl.uint32) << RANK_BITS) | initial_ranks.to(tl.uint32)
    initial_rank_priorities = tl.randint(seed=rank_priority_seed, offset=rank_random_offsets, n_rounds=10).to(tl.uint32)

    # [BT, BK]：以初选槽及其分数作为保底，允许保留初选槽，不强制换到其他副本。
    # 所有专家始终查询原始临时目标集合，不把其他 lane 的最终结果作为新锚点。
    best_physical_ids = initial_physical_ids
    best_priorities = initial_rank_priorities
    for partner_slot in tl.static_range(arg1=0, arg2=TOP_K):
        # [BT, 1]：同 token 第 partner_slot 个目标 rank 和分数，沿 top-k 列广播。
        partner_indices = tl.full(shape=(BLOCK_TOKENS, 1), value=partner_slot, dtype=tl.int32)
        partner_rank = tl.gather(src=initial_ranks, index=partner_indices, axis=1)
        partner_priority = tl.gather(src=initial_rank_priorities, index=partner_indices, axis=1)
        # [BT, BK]：每个专家在该目标 rank 上的副本；本地 lane 不查询远端表。
        candidate_physical_ids = tl.load(
            pointer=logical_to_rank_physical_map_ptr + rank_map_row_offsets + partner_rank,
            mask=remote_mask,
            other=-1,
        )
        # [BT, BK]：只在随机分数严格更小时更新；同分保留当前副本，不额外比较 rank/槽号。
        # 同分可能使 top-k 重排后选择不同物理副本，但不会改变 logical ID 或权重。
        better_priority = partner_priority < best_priorities
        use_candidate = remote_mask & (candidate_physical_ids >= 0) & better_priority
        best_physical_ids = tl.where(condition=use_candidate, x=candidate_physical_ids, y=best_physical_ids)
        best_priorities = tl.where(condition=use_candidate, x=partner_priority, y=best_priorities)
    # 本地 lane 始终保留初选本地槽；远端 lane 找不到更优副本时保留初选远端槽。
    tl.store(pointer=physical_topk_ids_ptr + topk_id_offsets, value=best_physical_ids, mask=valid_mask)

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
def eplb_hash_targets_topk_ids(
    logical_topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_to_rank_physical_map: torch.Tensor,
    current_rank: int,
    num_physical_experts_per_rank: int,
    route_counter: Optional[torch.Tensor] = None,
    update_route_counter: bool = False,
    route_recording_enabled: Optional[torch.Tensor] = None,
    node_world_size: int = 1,
) -> torch.Tensor:
    """本卡优先，远端 token 级伪随机初选后按共同 rank 优先级共址；logical ID 和权重不变。

    logical_topk_ids 为 [T, K]；复用 prefill 表 [E, 3 + W*S]，按 rank 查询表为 [E, W]。
    prefill 表第 0/1 列为 global_count/node_count，第 3 列起为有效 physical ID。
    表须按相同的 current_rank/node_world_size 构建；有本节点副本时在节点前缀中初选。
    node_world_size 默认 1（每节点一个 rank），仅用于拓扑校验；初选范围由表内 node_count 决定。
    两张表必须来自同一完整布局，每个专家至少有一个副本，每个 rank 内不可重复。
    初选使用 tl.randint(seed=current_rank, offset=token_index)，同 token 共享随机数。
    副本数相同的专家共享候选序号，但候选序号相同不保证目标 rank 相同。
    共址优先级也使用 Philox：seed 高位区分随机流，uint32 offset = (token_index << rank_bits) | rank。
    rank_bits = (world_size - 1).bit_length()，低位预留给 rank；非二次幂 world_size 同样适用。
    同 token 的专家共用 rank 随机分数；分数相同时保留当前副本，不额外比较 rank 或槽号。
    极少数同分情况下，top-k 重排可能改变物理副本选择，但 logical ID 与权重不变。
    token index 是本次输入的行号，不是请求 ID；改变 batch 排列可能改变初选与共址。
    伪随机数只提供统计意义的分散选择，不保证每批负载均匀或通信优于固定 owner。
    """
    assert logical_topk_ids.ndim == 2 and logical_topk_ids.is_contiguous()
    assert logical_topk_ids.dtype in (torch.int32, torch.int64)
    assert logical_topk_ids.shape[1] > 0
    assert logical_to_rank_physical_map.ndim == 2 and logical_to_rank_physical_map.is_contiguous()
    assert logical_to_physical_map.ndim == 2
    assert logical_to_physical_map.shape == (
        logical_to_rank_physical_map.shape[0],
        3 + logical_to_rank_physical_map.shape[1] * num_physical_experts_per_rank,
    )
    assert logical_to_physical_map.is_contiguous()
    assert logical_to_physical_map.dtype is torch.int32 and logical_to_rank_physical_map.dtype is torch.int32
    assert logical_to_physical_map.device == logical_to_rank_physical_map.device == logical_topk_ids.device
    assert 0 <= current_rank < logical_to_rank_physical_map.shape[1]
    assert 0 < node_world_size <= logical_to_rank_physical_map.shape[1]
    assert logical_to_rank_physical_map.shape[1] % node_world_size == 0
    assert num_physical_experts_per_rank > 0
    if update_route_counter:
        experts = logical_to_rank_physical_map.shape[0]
        assert route_counter is not None and route_counter.shape == (experts, experts)
        assert route_counter.dtype is torch.int64 and route_counter.is_contiguous()
        assert route_counter.device == logical_topk_ids.device
        assert route_recording_enabled is not None and route_recording_enabled.shape == (1,)
        assert route_recording_enabled.dtype is torch.int32 and route_recording_enabled.is_contiguous()
        assert route_recording_enabled.device == logical_topk_ids.device
        counter, enabled = route_counter, route_recording_enabled
    else:
        # 无统计时只传占位指针，constexpr 裁剪后不会访问这些表的统计位置。
        counter, enabled = logical_to_physical_map, logical_to_physical_map
    output = torch.empty_like(input=logical_topk_ids)
    if logical_topk_ids.numel() == 0:
        return output
    block_top_k = triton.next_power_of_2(n=logical_topk_ids.shape[1])
    block_tokens = max(1, 256 // block_top_k)
    _eplb_hash_targets_topk_ids_kernel[(triton.cdiv(x=logical_topk_ids.shape[0], y=block_tokens),)](
        logical_topk_ids_ptr=logical_topk_ids,
        physical_topk_ids_ptr=output,
        logical_to_physical_map_ptr=logical_to_physical_map,
        logical_to_physical_map_row_stride=logical_to_physical_map.stride(dim=0),
        logical_to_rank_physical_map_ptr=logical_to_rank_physical_map,
        logical_to_rank_physical_map_row_stride=logical_to_rank_physical_map.stride(dim=0),
        num_tokens=logical_topk_ids.shape[0],
        current_rank=current_rank,
        num_physical_experts_per_rank=num_physical_experts_per_rank,
        decode_route_counter_ptr=counter,
        decode_route_counter_row_stride=counter.stride(dim=0),
        decode_route_recording_enabled_ptr=enabled,
        UPDATE_ROUTE_COUNTER=update_route_counter,
        RANK_BITS=(logical_to_rank_physical_map.shape[1] - 1).bit_length(),
        TOP_K=logical_topk_ids.shape[1],
        BLOCK_TOKENS=block_tokens,
        BLOCK_TOP_K=block_top_k,
        num_warps=4,
        num_stages=1,
    )
    return output
