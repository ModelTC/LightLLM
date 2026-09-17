import torch
import triton
import triton.language as tl


@triton.jit
def _replica_index(token_index, logical_expert_id, num_valid_replicas):
    token_hash = token_index.to(tl.uint32) * 2654435769
    expert_hash = logical_expert_id.to(tl.uint32) * 2246822519
    return (token_hash + expert_hash) % num_valid_replicas.to(tl.uint32)


@triton.jit
def _eplb_repair_topk_ids_kernel(
    logical_topk_ids_ptr,
    physical_topk_ids_ptr,
    num_topk_ids,
    top_k,
    logical_to_physical_map_ptr,
    logical_to_physical_map_row_stride,
    logical_expert_counter_ptr,
    UPDATE_LOGICAL_EXPERT_COUNTER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 阶段 1：将二维 [num_tokens, top_k] 路由结果展平后分块处理。
    # topk_id_offsets 同时用于访问输入、输出，并可恢复它所属的 token 下标。
    topk_id_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_mask = topk_id_offsets < num_topk_ids
    logical_expert_ids = tl.load(logical_topk_ids_ptr + topk_id_offsets, mask=valid_mask, other=0)

    # 阶段 2：在 EPLB 采样窗口内，按 logical expert 统计本次路由负载。
    # 统计发生在 physical ID 修复之前，因此冗余副本不会拆散逻辑专家负载。
    if UPDATE_LOGICAL_EXPERT_COUNTER:
        tl.atomic_add(
            logical_expert_counter_ptr + logical_expert_ids,
            1,
            mask=valid_mask,
            sem="relaxed",
        )

    # 阶段 3：定位每个 logical expert 的打包映射行。第 0 列保存有效
    # physical 副本数，第 1 列标记当前 rank 是否有本地副本，第 2 列起
    # 保存可参与路由的 physical expert ID；本地副本存在时固定放在第一个槽。
    map_row_offsets = logical_expert_ids * logical_to_physical_map_row_stride
    num_valid_replicas = tl.load(logical_to_physical_map_ptr + map_row_offsets, mask=valid_mask, other=1)
    has_local_replica = tl.load(logical_to_physical_map_ptr + map_row_offsets + 1, mask=valid_mask, other=0)

    # 阶段 4：若当前 rank 持有该专家，强制选择第一个槽位以避免跨 rank
    # 通信；否则用 token 下标和 logical expert ID 生成稳定 hash，在有效
    # 副本范围内选择槽位，避免固定 token 位置长期偏向同一个副本。
    token_indices = topk_id_offsets // top_k
    hashed_replica_indices = _replica_index(token_indices, logical_expert_ids, num_valid_replicas)
    selected_replica_indices = tl.where(has_local_replica != 0, 0, hashed_replica_indices)

    # 阶段 5：读取选中槽位的 physical expert ID 并写入新的输出 tensor。
    # logical_topk_ids 只读，后续 callback 仍可安全观察原始逻辑路由结果。
    physical_expert_ids = tl.load(
        logical_to_physical_map_ptr + map_row_offsets + 2 + selected_replica_indices,
        mask=valid_mask,
        other=-1,
    )
    tl.store(physical_topk_ids_ptr + topk_id_offsets, physical_expert_ids, mask=valid_mask)


@torch.no_grad()
def eplb_repair_topk_ids(
    logical_topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_expert_counter: torch.Tensor,
    update_logical_expert_counter: bool,
) -> torch.Tensor:
    """将 logical top-k ID 转换为当前 EPLB 布局中的 physical expert ID。

    参数:
        logical_topk_ids: logical expert ID，shape 为 ``[num_tokens, top_k]``。
        logical_to_physical_map: 打包路由表，shape 为
            ``[num_logical_experts, 2 + routing_slots]``。第 0 列保存有效副本
            数量，第 1 列标记当前 rank 是否有本地副本，其余列保存 physical
            expert ID；存在本地副本时，其 physical ID 固定放在第一个槽位。
            有效副本之后未使用的 padding 槽位为 -1，kernel 不会读取它们。
        logical_expert_counter: 每个 logical expert 的累计路由次数，shape 为
            ``[num_logical_experts]``。
        update_logical_expert_counter: 是否将本次 logical 路由结果累计到
            ``logical_expert_counter``。

    返回:
        physical expert ID，shape 为 ``[num_tokens, top_k]``。
    """
    assert logical_topk_ids.is_contiguous()
    assert logical_topk_ids.ndim == 2
    assert logical_to_physical_map.ndim == 2
    assert logical_to_physical_map.shape[1] > 2
    assert logical_to_physical_map.stride(1) == 1
    assert logical_expert_counter.ndim == 1
    assert logical_expert_counter.shape[0] == logical_to_physical_map.shape[0]
    physical_topk_ids = torch.empty_like(logical_topk_ids)
    if logical_topk_ids.numel() == 0:
        return physical_topk_ids

    block_size = 512
    _eplb_repair_topk_ids_kernel[(triton.cdiv(logical_topk_ids.numel(), block_size),)](
        logical_topk_ids_ptr=logical_topk_ids,
        physical_topk_ids_ptr=physical_topk_ids,
        num_topk_ids=logical_topk_ids.numel(),
        top_k=logical_topk_ids.shape[1],
        logical_to_physical_map_ptr=logical_to_physical_map,
        logical_to_physical_map_row_stride=logical_to_physical_map.stride(0),
        logical_expert_counter_ptr=logical_expert_counter,
        UPDATE_LOGICAL_EXPERT_COUNTER=update_logical_expert_counter,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=1,
    )
    return physical_topk_ids
