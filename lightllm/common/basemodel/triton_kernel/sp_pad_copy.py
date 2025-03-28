# this kernel is used to padding token tensor,
# the padded tensor can be used to use sp policy.
# if token_num % sp_world_size != 0, will pad
# token to token_num % sp_world_size == 0, this.
# demo: token_num = 3, sp_world_size = 4, will get
# ans_tensor shape is [1, hidden_dim].
import torch

import triton
import triton.language as tl


@triton.jit
def _sp_pad_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    output_ptr,
    output_stride_0,
    output_stride_1,
    source_token_num,
    dest_token_start_index,
    dest_token_end_index,
    hidden_dim,
    grid_num,
    BLOCK_SIZE: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_SIZE)

    for dest_index in tl.range(dest_token_start_index + cur_index, dest_token_end_index, grid_num, num_stages=2):
        source_index = dest_index % source_token_num
        in_ptr = input_ptr + source_index * input_stride_0 + offs_d
        out_ptr = output_ptr + (dest_index - dest_token_start_index) * output_stride_0 + offs_d
        in_data = tl.load(in_ptr, mask=offs_d < hidden_dim, other=0.0)
        tl.store(out_ptr, in_data, mask=offs_d < hidden_dim)
    return


@torch.no_grad()
def sp_pad_copy(in_tensor: torch.Tensor, sp_rank_id: int, sp_world_size: int, alloc_func=torch.empty):
    assert in_tensor.is_contiguous()
    assert len(in_tensor.shape) == 2
    in_token_num, hidden_dim = in_tensor.shape

    if in_token_num % sp_world_size == 0:
        split_size = in_token_num // sp_world_size
        start = sp_rank_id * split_size
        end = start + split_size
        return in_tensor[start:end, :]

    out_token_num = triton.cdiv(in_token_num, sp_world_size) * sp_world_size // sp_world_size
    out_tensor = alloc_func((out_token_num, hidden_dim), dtype=in_tensor.dtype, device=in_tensor.device)
    out_token_start_index = out_token_num * sp_rank_id
    out_token_end_index = out_token_num * (sp_rank_id + 1)
    grid_num = max(64, triton.cdiv(out_token_num, 64))
    grid = (grid_num,)

    _sp_pad_kernel[grid](
        in_tensor,
        *in_tensor.stride(),
        out_tensor,
        *out_tensor.stride(),
        in_token_num,
        out_token_start_index,
        out_token_end_index,
        hidden_dim,
        grid_num,
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
        num_warps=1,
    )
    return out_tensor
