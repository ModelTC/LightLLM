"""Prefill 副本哈希分发：按 token 和逻辑专家 ID 在指定局部性范围内选择副本。"""

from typing import Optional

import torch
import triton
import triton.language as tl

from .prefill_eplb_route_statistics import _record_prefill_route_sample


@triton.jit
def _replica_index(token_index, logical_expert_id, num_valid_replicas):
    # 先用 logical expert ID 给 token index 加盐，再用 32-bit avalanche
    # finalizer 打散规律性 token 间隔，避免低位周期与副本数产生相关性。
    value = token_index.to(tl.uint32)
    value ^= (logical_expert_id.to(tl.uint32) + 1) * 0x9E3779B9
    value ^= value >> 16
    value *= 0x7FEB352D
    value ^= value >> 15
    value *= 0x846CA68B
    value ^= value >> 16
    value = value.to(tl.uint32)
    return value % num_valid_replicas.to(tl.uint32)


@triton.jit
def _eplb_prefill_replica_hash_topk_ids_kernel(
    logical_topk_ids_ptr,
    physical_topk_ids_ptr,
    num_topk_ids,
    logical_to_physical_map_ptr,
    logical_to_physical_map_row_stride,
    prefill_route_counter_ptr,
    prefill_route_counter_row_stride,
    prefill_route_sample_index_ptr,
    DISPATCH_MODE: tl.constexpr,
    UPDATE_PREFILL_ROUTE_COUNTER: tl.constexpr,
    NUM_LOGICAL_EXPERTS: tl.constexpr,
    PREFILL_ROUTE_COUNTER_CAPACITY: tl.constexpr,
    COUNTER_BLOCK_SIZE: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 阶段 1：将二维 [num_tokens, top_k] 路由结果展平后分块处理。
    # topk_id_offsets 同时用于访问输入、输出，并可恢复它所属的 token 下标。
    program_id = tl.program_id(0)
    topk_id_offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_mask = topk_id_offsets < num_topk_ids
    # 无效 logical / physical ID 统一填 -1；后续查表、写回和统计仍由 mask 屏蔽。
    logical_expert_ids = tl.load(logical_topk_ids_ptr + topk_id_offsets, mask=valid_mask, other=-1)

    # 阶段 2：定位每个 logical expert 的打包映射行。固定头部的布局为：
    #
    # [0] 所有 rank 上的有效副本总数
    # [1] 当前节点上的有效副本数，包含本卡副本
    # [2] 当前 GPU 上的有效副本数
    # [3:] 按本卡、本节点其他卡、其他节点排列的 physical expert IDs
    #
    # 因为三层候选在 physical ID 列表中都是连续前缀，所以选择对应层级的
    # count 后，可以直接对 [3:] 的前 count 项做 hash。
    map_row_offsets = logical_expert_ids * logical_to_physical_map_row_stride
    # 这里加载的是副本数量而非 expert ID：无效 lane 的全局数量填 1，
    # 本地数量填 0，使各分发模式最终以 1 取模，避免 hash 中出现模零。
    num_global_replicas = tl.load(logical_to_physical_map_ptr + map_row_offsets, mask=valid_mask, other=1)
    num_node_replicas = tl.load(logical_to_physical_map_ptr + map_row_offsets + 1, mask=valid_mask, other=0)
    num_current_gpu_replicas = tl.load(
        logical_to_physical_map_ptr + map_row_offsets + 2,
        mask=valid_mask,
        other=0,
    )

    # 阶段 3：根据调用方显式指定的 prefill 分发模式选择参与 hash 的候选前缀。
    if DISPATCH_MODE == 0:
        # current_gpu_first: 本卡 -> 全局。
        num_preferred_replicas = tl.where(
            num_current_gpu_replicas > 0,
            num_current_gpu_replicas,
            num_global_replicas,
        )
    elif DISPATCH_MODE == 1:
        # current_node_first: 本节点 -> 全局；不单独优先本卡。
        num_preferred_replicas = tl.where(
            num_node_replicas > 0,
            num_node_replicas,
            num_global_replicas,
        )
    else:
        # global_first: 直接在所有 rank 的有效副本间分发。
        num_preferred_replicas = num_global_replicas
    token_indices = topk_id_offsets // TOP_K
    selected_replica_indices = _replica_index(token_indices, logical_expert_ids, num_preferred_replicas)

    # 阶段 4：读取选中槽位的 physical expert ID 并写入新的输出 tensor。
    # logical_topk_ids 只读，后续 callback 仍可安全观察原始逻辑路由结果。
    physical_expert_ids = tl.load(
        logical_to_physical_map_ptr + map_row_offsets + 3 + selected_replica_indices,
        mask=valid_mask,
        other=-1,
    )
    tl.store(physical_topk_ids_ptr + topk_id_offsets, physical_expert_ids, mask=valid_mask)

    # 阶段 5：记录本次 prefill 路由采样。子函数负责目标行清零、program 间
    # ready 同步、logical expert 计数，以及最后完成者对 sample index 的提交。
    if UPDATE_PREFILL_ROUTE_COUNTER:
        _record_prefill_route_sample(
            logical_expert_ids,
            valid_mask,
            prefill_route_counter_ptr,
            prefill_route_counter_row_stride,
            prefill_route_sample_index_ptr,
            NUM_LOGICAL_EXPERTS,
            PREFILL_ROUTE_COUNTER_CAPACITY,
            COUNTER_BLOCK_SIZE,
        )


@torch.no_grad()
def eplb_prefill_replica_hash_topk_ids(
    logical_topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    prefill_route_counter: Optional[torch.Tensor],
    prefill_route_sample_index: Optional[torch.Tensor],
    update_prefill_route_counter: bool,
    mode: str,
) -> torch.Tensor:
    """按 prefill 分发策略将 logical top-k ID 映射为 physical expert ID。

    参数:
        logical_topk_ids: logical expert ID，shape 为 ``[num_tokens, top_k]``。
        logical_to_physical_map: 打包路由表，shape 为
            ``[num_logical_experts, 3 + routing_slots]``，单行布局为：

            ``[global_count, node_count, current_gpu_count, physical_ids..., padding...]``

            三个计数依次表示全局、当前节点和当前 GPU 上的有效副本数量。
            physical IDs 按本卡、本节点其他卡、其他节点排列；prefill 根据
            ``mode`` 使用本卡、当前节点或全局对应的连续候选前缀。有效副本之后
            未使用的 padding 槽位为 -1，kernel 不会读取它们。
        prefill_route_counter: prefill 路由采样的环形缓冲区，shape 为
            ``[sample_capacity, num_logical_experts]``。只在
            ``update_prefill_route_counter=True`` 时要求提供。
        prefill_route_sample_index: shape 为 ``[2]`` 的设备端同步状态。第 0 项是
            单调递增的 sample index；第 1 项用于核内清零与完成同步：0
            表示尚未清零，正数为 ready 标记 1 加上已完成的 program 数量；
            达到 ``num_programs + 1`` 后，最后完成者推进 sample index 并清零。
        update_prefill_route_counter: 是否记录本次 prefill 路由采样并推进 sample index。
        mode: 必须显式指定的 prefill 副本分发模式：

            * ``current_gpu_first``：本卡优先，没有本卡副本时回退到全局；
            * ``current_node_first``：本节点优先，没有节点内副本时回退到全局；
            * ``global_first``：直接在全局全部有效副本间分发。

    返回:
        physical expert ID，shape 为 ``[num_tokens, top_k]``。
    """
    dispatch_mode_ids = {
        "current_gpu_first": 0,
        "current_node_first": 1,
        "global_first": 2,
    }
    assert (
        mode in dispatch_mode_ids
    ), f"unsupported EPLB dispatch mode {mode!r}; expected one of {tuple(dispatch_mode_ids)}"
    dispatch_mode = dispatch_mode_ids[mode]

    assert logical_topk_ids.is_contiguous()
    assert logical_topk_ids.ndim == 2
    assert logical_to_physical_map.ndim == 2
    assert logical_to_physical_map.shape[1] > 3
    assert logical_to_physical_map.stride(1) == 1
    assert logical_topk_ids.device == logical_to_physical_map.device

    num_logical_experts = logical_to_physical_map.shape[0]
    if prefill_route_counter is not None:
        assert prefill_route_sample_index is not None
        assert prefill_route_counter.ndim == 2
        assert prefill_route_counter.shape[0] > 1
        assert prefill_route_counter.shape[1] == num_logical_experts
        assert prefill_route_counter.is_contiguous()
        assert prefill_route_counter.dtype is torch.int64
        assert prefill_route_counter.device == logical_topk_ids.device
        assert prefill_route_sample_index.shape == (2,)
        assert prefill_route_sample_index.dtype is torch.int64
        assert prefill_route_sample_index.is_contiguous()
        assert prefill_route_sample_index.device == prefill_route_counter.device
        kernel_prefill_route_counter = prefill_route_counter
        kernel_prefill_route_sample_index = prefill_route_sample_index
        prefill_route_counter_capacity = prefill_route_counter.shape[0]
    else:
        assert prefill_route_sample_index is None
        assert not update_prefill_route_counter
        # Triton pointer 不能传 None；constexpr 会裁掉未启用的统计分支。
        kernel_prefill_route_counter = logical_to_physical_map
        kernel_prefill_route_sample_index = logical_to_physical_map
        prefill_route_counter_capacity = 1

    physical_topk_ids = torch.empty_like(logical_topk_ids)
    if logical_topk_ids.numel() == 0:
        return physical_topk_ids

    block_size = 512
    _eplb_prefill_replica_hash_topk_ids_kernel[(triton.cdiv(logical_topk_ids.numel(), block_size),)](
        logical_topk_ids_ptr=logical_topk_ids,
        physical_topk_ids_ptr=physical_topk_ids,
        num_topk_ids=logical_topk_ids.numel(),
        logical_to_physical_map_ptr=logical_to_physical_map,
        logical_to_physical_map_row_stride=logical_to_physical_map.stride(0),
        prefill_route_counter_ptr=kernel_prefill_route_counter,
        prefill_route_counter_row_stride=kernel_prefill_route_counter.stride(0),
        prefill_route_sample_index_ptr=kernel_prefill_route_sample_index,
        DISPATCH_MODE=dispatch_mode,
        UPDATE_PREFILL_ROUTE_COUNTER=update_prefill_route_counter,
        NUM_LOGICAL_EXPERTS=num_logical_experts,
        PREFILL_ROUTE_COUNTER_CAPACITY=prefill_route_counter_capacity,
        COUNTER_BLOCK_SIZE=triton.next_power_of_2(num_logical_experts),
        TOP_K=logical_topk_ids.shape[1],
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=1,
    )
    return physical_topk_ids
