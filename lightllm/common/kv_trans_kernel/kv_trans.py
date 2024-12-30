import torch

import triton
import triton.language as tl


@triton.jit
def _kv_trans_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    input_token_idx_ptr,
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
        input_token_idx = tl.load(input_token_idx_ptr + tid)
        output_token_idx = tl.load(output_token_idx_ptr + tid)
        for block_idx in tl.range(0, tl.cdiv(head_num_dim, BLOCK_SIZE), 1, num_stages=NUM_STAGES):
            cur_offs = block_idx * BLOCK_SIZE + offs
            in_datas = tl.load(input_ptr + input_stride_0 * input_token_idx + cur_offs, mask=cur_offs < head_num_dim)
            tl.store(output_ptr + output_stride_0 * output_token_idx + cur_offs, in_datas, mask=cur_offs < head_num_dim)

        tid += grid_count

    return


def kv_trans(input: torch.Tensor, input_idx: torch.Tensor, output: torch.Tensor, output_idx: torch.Tensor):
    assert input.is_contiguous()
    assert output.is_contiguous()
    assert len(input.shape) == 3
    assert len(output.shape) == 3
    assert len(input_idx) == len(output_idx)

    _, head_num, head_dim = input.shape
    token_num = len(input_idx)
    # 用较少的资源来做数据传输，防止占用过多的 sm 计算单元
    grid_count = 20
    BLOCK_SIZE = 256
    NUM_STAGES = 3
    grid = (grid_count,)

    _kv_trans_kernel[grid](
        input,
        *input.stride(),
        input_idx,
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
