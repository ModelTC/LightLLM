"""Prefill 路由共用的逻辑专家环形采样，保留原有核内事务与同步规则。"""

import triton
import triton.language as tl


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
