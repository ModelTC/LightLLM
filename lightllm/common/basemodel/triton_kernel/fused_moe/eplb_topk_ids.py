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
def _record_prefill_route_sample(
    logical_expert_ids,
    valid_mask,
    prefill_route_counter_ptr,
    prefill_route_counter_row_stride,
    prefill_route_sample_index_ptr,
    NUM_LOGICAL_EXPERTS: tl.constexpr,
    PREFILL_ROUTE_COUNTER_CAPACITY: tl.constexpr,
    COUNTER_BLOCK_SIZE: tl.constexpr,
):
    """在主路由 kernel 内完成一次 prefill 采样事务。"""
    # 同步槽 ``prefill_route_sample_index[1]`` 的状态机：
    #
    #   0                  : 本次采样尚未清零；
    #   1                  : program 0 已清零，ready 标记已经发布；
    #   1 + completed      : ready 标记加上已经完成的 program 数；
    #   1 + num_programs   : 本次所有 program 均已完成。
    #
    # 所有调用必须排在同一 CUDA stream 上，不能并发复用同一组 counter
    # 和同步状态。
    program_id = tl.program_id(0)

    # 1. 所有 program 读取相同的 sample index，并通过取余定位本次写入的
    # 环形行。sample index 只有在全部 program 完成后才会推进，因此本次
    # kernel 生命周期内，各 program 看到的目标行保持不变。
    sample_index = tl.load(prefill_route_sample_index_ptr)
    sample_row = sample_index % PREFILL_ROUTE_COUNTER_CAPACITY

    if program_id == 0:
        # 2. program 0 负责初始化目标行。正常入口处同步槽必须为 0；清零
        # 完成后，通过 release 原子加一发布 ready=1，使此前的 store 对
        # 随后获得 ready 标记的其他 program 可见。
        sync_state = tl.atomic_add(
            prefill_route_sample_index_ptr + 1,
            0,
            sem="acquire",
            scope="gpu",
        )
        if sync_state == 0:
            expert_offsets = tl.arange(0, COUNTER_BLOCK_SIZE)
            tl.store(
                prefill_route_counter_ptr + sample_row * prefill_route_counter_row_stride + expert_offsets,
                0,
                mask=expert_offsets < NUM_LOGICAL_EXPERTS,
            )
            tl.atomic_add(
                prefill_route_sample_index_ptr + 1,
                1,
                sem="release",
                scope="gpu",
            )
    else:
        # 3. 其他 program 使用 acquire 原子读等待 ready 标记。只有观察到
        # sync >= 1 后才能离开循环，从而保证不会与 program 0 的清零 store
        # 并发访问同一个 counter 行。
        sync_state = tl.atomic_add(
            prefill_route_sample_index_ptr + 1,
            0,
            sem="acquire",
            scope="gpu",
        )
        while sync_state < 1:
            sync_state = tl.atomic_add(
                prefill_route_sample_index_ptr + 1,
                0,
                sem="acquire",
                scope="gpu",
            )

    # 4. 清零屏障通过后，各 program 将自己的 logical expert 路由结果原子
    # 累加到同一采样行。这里统计 logical expert，冗余 physical 副本不会
    # 拆散规划器观察到的负载信号。
    tl.atomic_add(
        prefill_route_counter_ptr + sample_row * prefill_route_counter_row_stride + logical_expert_ids,
        1,
        mask=valid_mask,
        sem="relaxed",
    )

    # 5. 本 program 完成计数后，向同步槽提交一个完成信号。调用发生在主
    # kernel 的 physical ID 写回之后，因此该信号同时表示两部分工作均完成。
    # atomic_add 返回旧值，故完成后的新值需要显式加一。当新值等于
    # ``num_programs + 1`` 时，ready 标记和全部 program 的完成信号均已到达。
    completed_programs = tl.atomic_add(
        prefill_route_sample_index_ptr + 1,
        1,
        sem="acq_rel",
        scope="gpu",
    )
    sync_after_completion = completed_programs + 1
    is_last_program = sync_after_completion == tl.num_programs(0) + 1
    if is_last_program:
        # 最后完成者提交本次事务：先推进 sample index，使下一次采样指向
        # 后续环形行；再把同步槽复位为 0，供下一次 program 0 执行清零。
        tl.atomic_add(
            prefill_route_sample_index_ptr,
            1,
            sem="release",
            scope="gpu",
        )
        tl.atomic_xchg(
            prefill_route_sample_index_ptr + 1,
            0,
            sem="release",
            scope="gpu",
        )


@triton.jit
def _eplb_repair_topk_ids_kernel(
    logical_topk_ids_ptr,
    physical_topk_ids_ptr,
    num_topk_ids,
    top_k,
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
    BLOCK_SIZE: tl.constexpr,
):
    # 阶段 1：将二维 [num_tokens, top_k] 路由结果展平后分块处理。
    # topk_id_offsets 同时用于访问输入、输出，并可恢复它所属的 token 下标。
    program_id = tl.program_id(0)
    topk_id_offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_mask = topk_id_offsets < num_topk_ids
    logical_expert_ids = tl.load(logical_topk_ids_ptr + topk_id_offsets, mask=valid_mask, other=0)

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
    num_global_replicas = tl.load(logical_to_physical_map_ptr + map_row_offsets, mask=valid_mask, other=1)
    num_node_replicas = tl.load(logical_to_physical_map_ptr + map_row_offsets + 1, mask=valid_mask, other=0)
    num_current_gpu_replicas = tl.load(
        logical_to_physical_map_ptr + map_row_offsets + 2,
        mask=valid_mask,
        other=0,
    )

    # 阶段 3：根据调用方显式指定的分发模式选择参与 hash 的候选前缀。
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
def eplb_repair_topk_ids(
    logical_topk_ids: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    prefill_route_counter: torch.Tensor,
    prefill_route_sample_index: torch.Tensor,
    update_prefill_route_counter: bool,
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
        prefill_route_counter: prefill 路由采样的环形缓冲区，shape 为
            ``[sample_capacity, num_logical_experts]``。一次 kernel 调用只写
            ``prefill_route_sample_index[0] % sample_capacity`` 对应的一行。
        prefill_route_sample_index: shape 为 ``[2]`` 的设备端同步状态。第 0 项
            是单调递增的 sample index；第 1 项用于核内清零与完成同步：0
            表示尚未清零，正数为 ready 标记 1 加上已完成的 program 数量；
            达到 ``num_programs + 1`` 后，最后完成者推进 sample index 并清零。
        update_prefill_route_counter: 是否记录本次 prefill 路由采样并推进 sample
            index。decode 或固定布局不需要采样时可以关闭。
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
    assert prefill_route_counter.ndim == 2
    assert prefill_route_counter.shape[0] > 1
    assert prefill_route_counter.shape[1] == logical_to_physical_map.shape[0]
    assert prefill_route_counter.is_contiguous()
    assert prefill_route_counter.dtype is torch.int64
    assert prefill_route_sample_index.shape == (2,)
    assert prefill_route_sample_index.dtype is torch.int64
    assert prefill_route_sample_index.device == prefill_route_counter.device
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
        prefill_route_counter_ptr=prefill_route_counter,
        prefill_route_counter_row_stride=prefill_route_counter.stride(0),
        prefill_route_sample_index_ptr=prefill_route_sample_index,
        DISPATCH_MODE=dispatch_mode,
        UPDATE_PREFILL_ROUTE_COUNTER=update_prefill_route_counter,
        NUM_LOGICAL_EXPERTS=prefill_route_counter.shape[1],
        PREFILL_ROUTE_COUNTER_CAPACITY=prefill_route_counter.shape[0],
        COUNTER_BLOCK_SIZE=triton.next_power_of_2(prefill_route_counter.shape[1]),
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=1,
    )
    return physical_topk_ids
