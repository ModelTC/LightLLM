import torch

import triton
import triton.language as tl


@triton.jit
def _kv_trans_prefill_node_kernel(
    input_mems_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    input_token_idx_ptr,
    input_dp_idx_ptr,
    output_ptr,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    output_token_idx_ptr,
    token_num: int,
    head_num: int,
    head_dim: int,
    grid_count: int,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)
    output_stride_1 = tl.cast(output_stride_1, dtype=tl.int64)

    head_num_dim = head_num * head_dim
    tid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    while tid < token_num:
        dp_index = tl.load(input_dp_idx_ptr + tid)
        input_token_idx = tl.load(input_token_idx_ptr + tid)
        output_token_idx = tl.load(output_token_idx_ptr + tid)
        input_ptr = tl.load(input_mems_ptr + dp_index).to(tl.pointer_type(output_ptr.dtype.element_ty))
        for block_idx in tl.range(0, tl.cdiv(head_num_dim, BLOCK_SIZE), 1, num_stages=NUM_STAGES):
            cur_offs = block_idx * BLOCK_SIZE + offs
            in_datas = tl.load(input_ptr + input_stride_0 * input_token_idx + cur_offs, mask=cur_offs < head_num_dim)
            tl.store(output_ptr + output_stride_0 * output_token_idx + cur_offs, in_datas, mask=cur_offs < head_num_dim)

        tid += grid_count

    return


def kv_trans_v2_for_p_node(
    input_mems: torch.Tensor,
    input_idx: torch.Tensor,
    input_dp_idx: torch.Tensor,
    output: torch.Tensor,
    output_idx: torch.Tensor,
    dp_size_in_node: int,
):
    """
    input_memes 是一个 torch.uint64 的tensor, 其内部存储了当前使用的对应的mem_manager对象中kv cache的首指针。
    """
    assert input_mems.is_contiguous()
    assert output.is_contiguous()
    assert len(input_mems.shape) == 1
    assert len(input_mems) == dp_size_in_node
    assert len(output.shape) == 3
    assert len(input_idx) == len(output_idx)
    assert len(input_idx) == len(input_dp_idx)

    _, head_num, head_dim = output.shape
    token_num = len(input_idx)
    # 用较少的资源来做数据传输，防止占用过多的 sm 计算单元
    grid_count = 20
    BLOCK_SIZE = 256
    NUM_STAGES = 3
    grid = (grid_count,)

    _kv_trans_prefill_node_kernel[grid](
        input_mems,
        *output.stride(),
        input_idx,
        input_dp_idx,
        output,
        *output.stride(),
        output_idx,
        token_num=token_num,
        head_num=head_num,
        head_dim=head_dim,
        grid_count=grid_count,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_STAGES=NUM_STAGES,
        num_warps=1,
    )
    return


@triton.jit
def _kv_trans_decode_node_kernel(
    output_mems_ptr,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    output_token_idx_ptr,
    output_token_dp_index_ptr,
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    input_token_idx_ptr,
    token_num: int,
    head_num: int,
    head_dim: int,
    grid_count: int,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    CARD_NUM_PER_D: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)
    output_stride_1 = tl.cast(output_stride_1, dtype=tl.int64)

    head_num_dim = head_num * head_dim
    tid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    while tid < token_num:
        dp_index = tl.load(output_token_dp_index_ptr + tid)
        input_token_idx = tl.load(input_token_idx_ptr + tid)
        output_token_idx = tl.load(output_token_idx_ptr + tid)
        for block_idx in tl.range(0, tl.cdiv(head_num_dim, BLOCK_SIZE), 1, num_stages=NUM_STAGES):
            cur_offs = block_idx * BLOCK_SIZE + offs
            in_datas = tl.load(input_ptr + input_stride_0 * input_token_idx + cur_offs, mask=cur_offs < head_num_dim)
            for mem_index in tl.range(
                dp_index * CARD_NUM_PER_D, (dp_index + 1) * CARD_NUM_PER_D, num_stages=NUM_STAGES
            ):
                output_ptr = tl.load(output_mems_ptr + mem_index).to(tl.pointer_type(input_ptr.dtype.element_ty))
                tl.store(
                    output_ptr + output_stride_0 * output_token_idx + cur_offs, in_datas, mask=cur_offs < head_num_dim
                )

        tid += grid_count

    return


def kv_trans_v2_for_d_node(
    output_mems: torch.Tensor,
    output_idx: torch.Tensor,
    output_dp_idx: torch.Tensor,
    input: torch.Tensor,
    input_idx: torch.Tensor,
    dp_size_in_node: int,
):
    """
    output_mems 是一个 torch.uint64 的tensor, 其内部存储了当前使用的对应的mem_manager对象中kv cache的首指针。
    """
    assert output_mems.is_contiguous()
    assert input.is_contiguous()
    assert len(output_mems.shape) == 1
    assert len(input.shape) == 3
    assert len(input_idx) == len(output_idx)
    assert len(input_idx) == len(output_dp_idx)
    assert len(output_mems) % dp_size_in_node == 0

    card_num_per_d = len(output_mems) // dp_size_in_node

    _, head_num, head_dim = input.shape
    token_num = len(input_idx)
    # 用较少的资源来做数据传输，防止占用过多的 sm 计算单元
    grid_count = 20
    BLOCK_SIZE = 256
    NUM_STAGES = 3
    grid = (grid_count,)

    _kv_trans_decode_node_kernel[grid](
        output_mems,
        *input.stride(),
        output_idx,
        output_dp_idx,
        input,
        *input.stride(),
        input_idx,
        token_num=token_num,
        head_num=head_num,
        head_dim=head_dim,
        grid_count=grid_count,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_STAGES=NUM_STAGES,
        CARD_NUM_PER_D=card_num_per_d,
        num_warps=1,
    )
    return


@triton.jit
def _kv_trans_for_dp_kernel(
    input_mems_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    input_stride_3,
    input_token_idx_ptr,
    input_token_dp_index_ptr,
    output_ptr,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    output_stride_3,
    output_token_idx_ptr,
    layer_num: tl.constexpr,
    token_num: int,
    head_num: int,
    head_dim: int,
    grid_count: int,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    CARD_NUM_PER_D: tl.constexpr,
    RANK_IN_DP: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)
    output_stride_1 = tl.cast(output_stride_1, dtype=tl.int64)

    head_num_dim = head_num * head_dim
    tid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    while tid < token_num:
        dp_index = tl.load(input_token_dp_index_ptr + tid)
        mem_index = RANK_IN_DP + dp_index * CARD_NUM_PER_D
        input_token_idx = tl.load(input_token_idx_ptr + tid)
        output_token_idx = tl.load(output_token_idx_ptr + tid)

        input_ptr = tl.load(input_mems_ptr + mem_index).to(tl.pointer_type(output_ptr.dtype.element_ty))
        for layer_idx in tl.range(0, layer_num, 1):
            for block_idx in tl.range(0, tl.cdiv(head_num_dim, BLOCK_SIZE), 1, num_stages=NUM_STAGES):
                cur_offs = block_idx * BLOCK_SIZE + offs
                in_datas = tl.load(
                    input_ptr + input_stride_0 * layer_idx + input_stride_1 * input_token_idx + cur_offs,
                    mask=cur_offs < head_num_dim,
                )
                tl.store(
                    output_ptr + output_stride_0 * layer_idx + output_stride_1 * output_token_idx + cur_offs,
                    in_datas,
                    mask=cur_offs < head_num_dim,
                )

        tid += grid_count

    return


def kv_trans_for_dp(
    input_mems: torch.Tensor,
    input_idx: torch.Tensor,
    input_dp_idx: torch.Tensor,
    output: torch.Tensor,
    output_idx: torch.Tensor,
    dp_size_in_node: int,
    rank_in_dp: int,
):
    """
    input_mems 是一个 torch.uint64 的tensor, shape为(layer_num, mem_num)，其内部存储了当前使用的对应的mem_manager对象中kv cache的首指针。
    """
    assert input_mems.is_contiguous()
    assert output.is_contiguous()
    assert len(input_mems.shape) == 1
    assert len(output.shape) == 4
    assert len(input_idx) == len(output_idx)
    assert len(output_idx) == len(input_dp_idx)
    assert len(input_mems) % dp_size_in_node == 0

    card_num_per_d = len(input_mems) // dp_size_in_node

    layer_num, _, head_num, head_dim = output.shape
    token_num = len(output_idx)
    # 用较少的资源来做数据传输，防止占用过多的 sm 计算单元
    grid_count = 20
    BLOCK_SIZE = 256
    NUM_STAGES = 3
    grid = (grid_count,)

    _kv_trans_for_dp_kernel[grid](
        input_mems,
        *output.stride(),
        input_idx,
        input_dp_idx,
        output,
        *output.stride(),
        output_idx,
        layer_num=layer_num,
        token_num=token_num,
        head_num=head_num,
        head_dim=head_dim,
        grid_count=grid_count,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_STAGES=NUM_STAGES,
        CARD_NUM_PER_D=card_num_per_d,
        RANK_IN_DP=rank_in_dp,
        num_warps=1,
    )

    return


@triton.jit
def _kv_trans_prefill_node_batch_kernel(
    source_pools,
    destination_pools,
    slot_bytes,
    tasks,
    blocks,
    block_num,
    LAYERS: tl.constexpr,
    DP_SIZE: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    # Each task owns its index arrays. Copy all layer layouts without expanding
    # a source-rank array per token or launching one kernel per layer.
    work = tl.program_id(0)
    rows = tl.arange(0, 32)
    cols = tl.arange(0, BLOCK_BYTES)
    while work < block_num * LAYERS:
        layer = work % LAYERS
        block = work // LAYERS
        task = tl.load(blocks + block * 2)
        start = tl.load(blocks + block * 2 + 1) * 256
        source_dp = tl.load(tasks + task * 4)
        src_indexes = tl.load(tasks + task * 4 + 1).to(tl.pointer_type(tl.int32))
        dst_indexes = tl.load(tasks + task * 4 + 2).to(tl.pointer_type(tl.int32))
        length = tl.load(tasks + task * 4 + 3)
        width = tl.load(slot_bytes + layer).to(tl.int64)
        source = tl.load(source_pools + layer * DP_SIZE + source_dp).to(tl.pointer_type(tl.uint8))
        destination = tl.load(destination_pools + layer).to(tl.pointer_type(tl.uint8))
        for offset in range(0, 256, 32):
            indexes = start + offset + rows
            src = tl.load(src_indexes + indexes, mask=indexes < length, other=0).to(tl.int64)
            dst = tl.load(dst_indexes + indexes, mask=indexes < length, other=0).to(tl.int64)
            mask = (indexes[:, None] < length) & (cols[None, :] < width)
            values = tl.load(source + src[:, None] * width + cols[None, :], mask=mask, other=0)
            tl.store(destination + dst[:, None] * width + cols[None, :], values, mask=mask)
        work += tl.num_programs(0)


def kv_trans_v2_for_p_node_batch(source_pools, destination_pools, slot_bytes, tasks, blocks, max_slot_bytes):
    """Copy byte-addressed token slots for a batch of requests and layer layouts.

    source_pools is [layer, DP], destination_pools/slot_bytes are [layer].
    tasks is uint64 [task, (source DP, int32 source-index pointer,
    int32 destination-index pointer, length)]; blocks is [task, 256-token block].
    The caller retains pools and index arrays until the stream completes.
    """
    layers, dp_size = source_pools.shape
    assert destination_pools.numel() == slot_bytes.numel() == layers
    assert tasks.shape[1] == 4 and blocks.shape[1] == 2
    _kv_trans_prefill_node_batch_kernel[(20,)](
        source_pools,
        destination_pools,
        slot_bytes,
        tasks,
        blocks,
        blocks.shape[0],
        LAYERS=layers,
        DP_SIZE=dp_size,
        BLOCK_BYTES=triton.next_power_of_2(max_slot_bytes),
        num_warps=4,
    )
