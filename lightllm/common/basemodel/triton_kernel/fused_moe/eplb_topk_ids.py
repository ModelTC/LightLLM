import torch
import triton
import triton.language as tl


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
def _eplb_repair_topk_ids_kernel(
    logical_topk_ids_ptr,
    physical_topk_ids_ptr,
    num_topk_ids,
    top_k,
    logical_to_physical_map_ptr,
    logical_to_physical_map_row_stride,
    logical_expert_counter_ptr,
    DISPATCH_MODE: tl.constexpr,
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

    # 阶段 3：定位每个 logical expert 的打包映射行。固定头部的布局为：
    #
    # [0] 所有 rank 上的有效副本总数
    # [1] 当前节点上的有效副本数，包含本卡副本
    # [2] 当前 GPU 上的有效副本数
    # [3:] 按本卡、本节点其他卡、其他节点排列的 physical expert IDs
    #
    # 因为三层候选在 physical ID 列表中都是连续前缀，所以选择对应层级的
    # count 后，可以直接对 [3:] 的前 count 项做 hash。
    map_row_offsets = logical_expert_ids * logical_to_physical_map_row_stride
    num_global_replicas = tl.load(logical_to_physical_map_ptr + map_row_offsets, mask=valid_mask, other=1)
    num_node_replicas = tl.load(logical_to_physical_map_ptr + map_row_offsets + 1, mask=valid_mask, other=0)
    num_current_gpu_replicas = tl.load(
        logical_to_physical_map_ptr + map_row_offsets + 2,
        mask=valid_mask,
        other=0,
    )

    # 阶段 4：根据调用方显式指定的分发模式选择参与 hash 的候选前缀。
    if DISPATCH_MODE == 0:
        # current_gpu_first: 本卡 -> 全局。
        # TODO: 等 EPLB 布局算法支持节点拓扑感知后，再考虑增加
        # 本卡 -> 本节点 -> 全局的分层回退行为。
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
    token_indices = topk_id_offsets // top_k
    selected_replica_indices = _replica_index(token_indices, logical_expert_ids, num_preferred_replicas)

    # 阶段 5：读取选中槽位的 physical expert ID 并写入新的输出 tensor。
    # logical_topk_ids 只读，后续 callback 仍可安全观察原始逻辑路由结果。
    physical_expert_ids = tl.load(
        logical_to_physical_map_ptr + map_row_offsets + 3 + selected_replica_indices,
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
    mode: str,
) -> torch.Tensor:
    """将 logical top-k ID 转换为当前 EPLB 布局中的 physical expert ID。

    参数:
        logical_topk_ids: logical expert ID，shape 为 ``[num_tokens, top_k]``。
        logical_to_physical_map: 打包路由表，shape 为
            ``[num_logical_experts, 3 + routing_slots]``，单行布局为：

            ``[global_count, node_count, current_gpu_count, physical_ids..., padding...]``

            三个计数依次表示全局、当前节点和当前 GPU 上的有效副本数量。
            physical IDs 按本卡、本节点其他卡、其他节点排列；具体参与分发的
            候选前缀由 ``mode`` 决定。
            有效副本之后未使用的 padding 槽位为 -1，kernel 不会读取它们。
        logical_expert_counter: 每个 logical expert 的累计路由次数，shape 为
            ``[num_logical_experts]``。
        update_logical_expert_counter: 是否将本次 logical 路由结果累计到
            ``logical_expert_counter``。固定布局不需要动态重排时可以关闭。
        mode: 必须显式指定的副本分发模式，不提供默认值：

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
        DISPATCH_MODE=dispatch_mode,
        UPDATE_LOGICAL_EXPERT_COUNTER=update_logical_expert_counter,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=1,
    )
    return physical_topk_ids
