import torch
import triton
import triton.language as tl


@triton.jit
def _repeat_rope_tensor(
    in_ptr,
    in_stride_0,
    in_stride_1,
    in_stride_2,
    out_ptr,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    copy_head_num,
    head_dim,
    total_len,
    BLOCK_N: tl.constexpr,
):
    start_index = tl.program_id(0)
    grid_num = tl.num_programs(0)

    offs_d = tl.arange(0, BLOCK_N)
    for cur_index in range(start_index, total_len, step=grid_num):
        in_tensor = tl.load(
            in_ptr + in_stride_0 * cur_index + in_stride_1 * 0 + offs_d, mask=offs_d < head_dim, other=0
        )
        for cur_head in tl.range(copy_head_num, num_stages=3):
            tl.store(
                out_ptr + out_stride_0 * cur_index + out_stride_1 * cur_head + offs_d, in_tensor, mask=offs_d < head_dim
            )
    return


@torch.no_grad()
def repeat_rope(dest_tensor: torch.Tensor, source_tensor: torch.Tensor):
    assert source_tensor.stride(2) == 1 and dest_tensor.stride(2) == 1
    repeat_head_num = dest_tensor.shape[1]
    seq_len, head_num, head_dim = source_tensor.shape
    assert head_num == 1

    BLOCK_N = triton.next_power_of_2(head_dim)

    if BLOCK_N <= 256:
        num_warps = 1
    elif BLOCK_N <= 1024:
        num_warps = 4
    else:
        num_warps = 8

    if seq_len <= 16 * 1024:
        grid = (seq_len,)
    else:
        grid = (16 * 1024,)

    _repeat_rope_tensor[grid](
        source_tensor,
        *source_tensor.stride(),
        dest_tensor,
        *dest_tensor.stride(),
        repeat_head_num,
        head_dim,
        seq_len,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return
