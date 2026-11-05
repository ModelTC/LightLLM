"""Decode 小批量分发：每个逻辑专家固定使用最小映射中的唯一物理槽。"""

from typing import Optional

import torch
import triton
import triton.language as tl

from .decode_eplb_route_statistics import _record_decode_route_cooccurrence


@triton.jit
def _eplb_minimal_topk_ids_kernel(
    logical_topk_ids_ptr,
    physical_topk_ids_ptr,
    minimal_logical_to_physical_map_ptr,
    decode_route_counter_ptr,
    decode_route_recording_enabled_ptr,
    num_tokens,
    decode_route_counter_row_stride,
    UPDATE_ROUTE_COUNTER: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_TOP_K: tl.constexpr,
):
    token_offsets = tl.program_id(axis=0) * BLOCK_TOKENS + tl.arange(start=0, end=BLOCK_TOKENS)  # [BT]：token 行号。
    topk_slots = tl.arange(start=0, end=BLOCK_TOP_K)  # [BK]：补齐后的 top-k 列号。
    topk_id_offsets = token_offsets[:, None] * TOP_K + topk_slots[None, :]  # [BT, BK]：输入/输出展平位置。
    valid_mask = (token_offsets[:, None] < num_tokens) & (topk_slots[None, :] < TOP_K)  # [BT, BK]：有效 token/expert。
    logical_expert_ids = tl.load(  # [BT, BK]：逻辑专家 ID；补齐位置为 -1。
        pointer=logical_topk_ids_ptr + topk_id_offsets,
        mask=valid_mask,
        other=-1,
    )
    # [BT, BK]：直接查询唯一物理槽，不做本卡优先或副本共址，避免激活更多物理专家。
    physical_topk_ids = tl.load(
        pointer=minimal_logical_to_physical_map_ptr + logical_expert_ids,
        mask=valid_mask,
        other=-1,
    )
    tl.store(pointer=physical_topk_ids_ptr + topk_id_offsets, value=physical_topk_ids, mask=valid_mask)

    # 共现仍按原始 logical ID 记录；映射和采样在同一个 kernel 中完成。
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
def eplb_minimal_topk_ids(
    logical_topk_ids: torch.Tensor,
    minimal_logical_to_physical_map: torch.Tensor,
    route_counter: Optional[torch.Tensor] = None,
    update_route_counter: bool = False,
    route_recording_enabled: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """直接将 [T,K] logical ID 映射到 [E] 表中的唯一物理槽，保持顺序和 dtype。

    映射表由布局生成，必须覆盖全部逻辑专家；所有源 rank 使用相同的最小映射。
    即使本卡有其他副本，也使用表中指定的槽位，不参与 rank 加权评分。
    统计开关和映射表均可原地更新，CUDA Graph 重放不需要重新捕获。
    """
    assert logical_topk_ids.ndim == 2 and logical_topk_ids.is_contiguous()
    assert logical_topk_ids.dtype in (torch.int32, torch.int64)
    assert logical_topk_ids.numel() > 0, "logical_topk_ids must be non-empty"
    assert minimal_logical_to_physical_map.ndim == 1 and minimal_logical_to_physical_map.is_contiguous()
    assert minimal_logical_to_physical_map.dtype is torch.int32
    assert logical_topk_ids.device == minimal_logical_to_physical_map.device
    if update_route_counter:
        experts = minimal_logical_to_physical_map.shape[0]
        assert route_counter is not None and route_counter.shape == (experts, experts)
        assert route_counter.dtype is torch.int64 and route_counter.is_contiguous()
        assert route_recording_enabled is not None and route_recording_enabled.shape == (1,)
        assert route_recording_enabled.dtype is torch.int32 and route_recording_enabled.is_contiguous()
        assert route_counter.device == logical_topk_ids.device == route_recording_enabled.device
        counter, enabled = route_counter, route_recording_enabled
    else:
        # 不记录统计时只传占位指针，constexpr 裁剪后不会访问它们。
        counter, enabled = minimal_logical_to_physical_map, minimal_logical_to_physical_map
    block_top_k = triton.next_power_of_2(n=logical_topk_ids.shape[1])
    block_tokens = max(1, 256 // block_top_k)
    output = torch.empty_like(input=logical_topk_ids)
    _eplb_minimal_topk_ids_kernel[(triton.cdiv(x=logical_topk_ids.shape[0], y=block_tokens),)](
        logical_topk_ids_ptr=logical_topk_ids,
        physical_topk_ids_ptr=output,
        minimal_logical_to_physical_map_ptr=minimal_logical_to_physical_map,
        decode_route_counter_ptr=counter,
        decode_route_recording_enabled_ptr=enabled,
        num_tokens=logical_topk_ids.shape[0],
        decode_route_counter_row_stride=counter.stride(dim=0),
        UPDATE_ROUTE_COUNTER=update_route_counter,
        TOP_K=logical_topk_ids.shape[1],
        BLOCK_TOKENS=block_tokens,
        BLOCK_TOP_K=block_top_k,
        num_warps=4,
        num_stages=1,
    )
    return output
