"""Prefill 小批量最小映射：每个逻辑专家固定使用一个物理槽，保留逻辑负载采样。"""

from typing import Optional

import torch
import triton
import triton.language as tl

from .prefill_eplb_route_statistics import _record_prefill_route_sample


@triton.jit
def _eplb_prefill_minimal_topk_ids_kernel(
    logical_topk_ids_ptr,
    physical_topk_ids_ptr,
    num_topk_ids,
    minimal_logical_to_physical_map_ptr,
    prefill_route_counter_ptr,
    prefill_route_counter_row_stride,
    prefill_route_sample_index_ptr,
    UPDATE_PREFILL_ROUTE_COUNTER: tl.constexpr,
    NUM_LOGICAL_EXPERTS: tl.constexpr,
    PREFILL_ROUTE_COUNTER_CAPACITY: tl.constexpr,
    COUNTER_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # [B]：把 [T, K] 展平处理，mask 同时屏蔽输入、映射查表和输出中的补齐 lane。
    topk_id_offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(start=0, end=BLOCK_SIZE)
    valid_mask = topk_id_offsets < num_topk_ids
    logical_expert_ids = tl.load(pointer=logical_topk_ids_ptr + topk_id_offsets, mask=valid_mask, other=-1)
    # [B]：所有源 rank 共用相同的 [E] 映射，不再激活该专家的其他冗余槽。
    physical_expert_ids = tl.load(
        pointer=minimal_logical_to_physical_map_ptr + logical_expert_ids,
        mask=valid_mask,
        other=-1,
    )
    tl.store(pointer=physical_topk_ids_ptr + topk_id_offsets, value=physical_expert_ids, mask=valid_mask)

    # 仍按原始 logical ID 采样；清零、核内同步与 sample index 提交与哈希路径一致。
    if UPDATE_PREFILL_ROUTE_COUNTER:
        _record_prefill_route_sample(
            logical_expert_ids=logical_expert_ids,
            valid_mask=valid_mask,
            prefill_route_counter_ptr=prefill_route_counter_ptr,
            prefill_route_counter_row_stride=prefill_route_counter_row_stride,
            prefill_route_sample_index_ptr=prefill_route_sample_index_ptr,
            NUM_LOGICAL_EXPERTS=NUM_LOGICAL_EXPERTS,
            PREFILL_ROUTE_COUNTER_CAPACITY=PREFILL_ROUTE_COUNTER_CAPACITY,
            COUNTER_BLOCK_SIZE=COUNTER_BLOCK_SIZE,
        )


@torch.no_grad()
def eplb_prefill_minimal_topk_ids(
    logical_topk_ids: torch.Tensor,
    minimal_logical_to_physical_map: torch.Tensor,
    prefill_route_counter: Optional[torch.Tensor],
    prefill_route_sample_index: Optional[torch.Tensor],
    update_prefill_route_counter: bool,
) -> torch.Tensor:
    """直接查 [E] 最小映射，返回相同 shape/dtype 的 physical ID，不修改输入。"""
    assert logical_topk_ids.ndim == 2
    assert logical_topk_ids.dtype in (torch.int32, torch.int64)
    assert logical_topk_ids.is_contiguous()
    assert minimal_logical_to_physical_map.ndim == 1
    assert minimal_logical_to_physical_map.dtype is torch.int32
    assert minimal_logical_to_physical_map.is_contiguous()
    assert minimal_logical_to_physical_map.device == logical_topk_ids.device

    num_logical_experts = minimal_logical_to_physical_map.shape[0]
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
        kernel_counter = prefill_route_counter
        kernel_sample_index = prefill_route_sample_index
        counter_capacity = prefill_route_counter.shape[0]
    else:
        assert prefill_route_sample_index is None
        assert not update_prefill_route_counter
        # Triton pointer 不能传 None；constexpr 会裁掉未启用的统计分支。
        kernel_counter = minimal_logical_to_physical_map
        kernel_sample_index = minimal_logical_to_physical_map
        counter_capacity = 1

    output = torch.empty_like(logical_topk_ids)
    # 保留 prefill 空输入行为：不启动 kernel，也不推进环形采样位置。
    if logical_topk_ids.numel() == 0:
        return output

    block_size = 512
    _eplb_prefill_minimal_topk_ids_kernel[(triton.cdiv(x=logical_topk_ids.numel(), y=block_size),)](
        logical_topk_ids_ptr=logical_topk_ids,
        physical_topk_ids_ptr=output,
        num_topk_ids=logical_topk_ids.numel(),
        minimal_logical_to_physical_map_ptr=minimal_logical_to_physical_map,
        prefill_route_counter_ptr=kernel_counter,
        prefill_route_counter_row_stride=kernel_counter.stride(dim=0),
        prefill_route_sample_index_ptr=kernel_sample_index,
        UPDATE_PREFILL_ROUTE_COUNTER=update_prefill_route_counter,
        NUM_LOGICAL_EXPERTS=num_logical_experts,
        PREFILL_ROUTE_COUNTER_CAPACITY=counter_capacity,
        COUNTER_BLOCK_SIZE=triton.next_power_of_2(n=num_logical_experts),
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=1,
    )
    return output
