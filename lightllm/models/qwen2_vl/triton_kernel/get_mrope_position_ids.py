import torch
import triton
import triton.language as tl


@triton.jit
def _get_mrope_position_triton(
    b_image_start_idx: torch.Tensor,
    b_image_pos_delta: torch.Tensor,
    b_image_nums: torch.Tensor,
    b_image_start_num: torch.Tensor,
    b_image_len: torch.Tensor,
    b_image_cu_len: torch.Tensor,
    b_image_position_id: torch.Tensor,
    b_image_position_id_stride0: torch.Tensor,
    position_ids: torch.Tensor,
    position_ids_stride0: torch.Tensor,
    b_ready_cache_len: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_start_loc: torch.Tensor,
    BLOCK_SIZE: tl.constexpr,
) -> torch.Tensor:
    cur_batch = tl.program_id(0)
    if cur_batch == 0:
        return
    cache_len = tl.load(b_ready_cache_len + cur_batch)
    seq_len = tl.load(b_seq_len + cur_batch)
    image_num = tl.load(b_image_nums + cur_batch)
    image_start_num = tl.load(b_image_start_num + cur_batch)
    start_loc = tl.load(b_start_loc + cur_batch)
    for i in range(image_num):
        local_image_start_idx = tl.load(b_image_start_idx + image_start_num + i)
        image_start_idx = start_loc + local_image_start_idx
        image_len = tl.load(b_image_len + image_start_num + i)
        cu_image_len = tl.load(b_image_cu_len + image_start_num + i)
        for j in range(0, image_len, BLOCK_SIZE):
            off = j + tl.arange(0, BLOCK_SIZE)
            t_pos = (
                tl.load(b_image_position_id + off + cu_image_len, mask=off < image_len, other=0.0)
                + local_image_start_idx
            )
            h_pos = (
                tl.load(
                    b_image_position_id + b_image_position_id_stride0 + off + cu_image_len,
                    mask=off < image_len,
                    other=0.0,
                )
                + local_image_start_idx
            )
            w_pos = (
                tl.load(
                    b_image_position_id + b_image_position_id_stride0 * 2 + off + cu_image_len,
                    mask=off < image_len,
                    other=0.0,
                )
                + local_image_start_idx
            )
            tl.store(
                position_ids + off + image_start_idx, t_pos, mask=(off < image_len) & (local_image_start_idx + off >= 0)
            )
            tl.store(
                position_ids + position_ids_stride0 + off + image_start_idx,
                h_pos,
                mask=(off < image_len) & (local_image_start_idx + off >= 0),
            )
            tl.store(
                position_ids + position_ids_stride0 * 2 + off + image_start_idx,
                w_pos,
                mask=(off < image_len) & (local_image_start_idx + off >= 0),
            )
            t_pos = tl.load(position_ids + off + image_start_idx, mask=(off < seq_len), other=0.0)

    for i in range(image_num):
        local_image_start_idx = tl.load(b_image_start_idx + image_start_num + i)
        image_len = tl.load(b_image_len + image_start_num + i)
        image_delta = tl.load(b_image_pos_delta + image_start_num + i)
        image_end = local_image_start_idx + image_len - cache_len
        for j in range(image_end, seq_len, BLOCK_SIZE):
            off = j + tl.arange(0, BLOCK_SIZE)
            t_pos = tl.load(position_ids + off + start_loc, mask=(off < seq_len), other=0.0) + image_delta
            h_pos = (
                tl.load(position_ids + position_ids_stride0 + off + start_loc, mask=(off < seq_len), other=0.0)
                + image_delta
            )
            w_pos = (
                tl.load(position_ids + position_ids_stride0 * 2 + off + start_loc, mask=(off < seq_len), other=0.0)
                + image_delta
            )
            tl.store(position_ids + off + start_loc, t_pos, mask=(off < seq_len))
            tl.store(position_ids + position_ids_stride0 + off + start_loc, h_pos, mask=(off < seq_len))
            tl.store(position_ids + position_ids_stride0 * 2 + off + start_loc, w_pos, mask=(off < seq_len))
    return


def get_mrope_position_triton(
    b_image_start_idx: torch.Tensor,
    b_image_pos_delta: torch.Tensor,
    b_image_nums: torch.Tensor,
    b_image_start_num: torch.Tensor,
    b_image_len: torch.Tensor,
    b_image_cu_len: torch.Tensor,
    b_image_position_id: torch.Tensor,
    position_ids: torch.Tensor,
    b_ready_cache_len: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_start_loc: torch.Tensor,
) -> torch.Tensor:

    batch_size = b_seq_len.shape[0]
    grid = (batch_size,)
    BLOCK_SIZE = 64
    _get_mrope_position_triton[grid](
        b_image_start_idx=b_image_start_idx,
        b_image_pos_delta=b_image_pos_delta,
        b_image_nums=b_image_nums,
        b_image_start_num=b_image_start_num,
        b_image_len=b_image_len,
        b_image_cu_len=b_image_cu_len,
        b_image_position_id=b_image_position_id,
        b_image_position_id_stride0=b_image_position_id.stride(0),
        position_ids=position_ids,
        position_ids_stride0=position_ids.stride(0),
        b_ready_cache_len=b_ready_cache_len,
        b_seq_len=b_seq_len,
        b_start_loc=b_start_loc,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def test():
    b_image_start_idx = torch.tensor([0, 0, 2], dtype=torch.int32, device="cuda")
    b_image_pos_delta = torch.tensor([-1, -2, -2], dtype=torch.int32, device="cuda")
    b_image_nums = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    b_image_start_num = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    b_image_len = torch.tensor([3, 2, 2], dtype=torch.int32, device="cuda")
    b_image_cu_len = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
    b_image_position_id = torch.tensor(
        [[0, 0, 0, 0, 0, 0, 0], [11, 11, 11, 21, 21, 31, 31], [12, 12, 12, 22, 22, 32, 32]],
        dtype=torch.int32,
        device="cuda",
    )
    position_ids = (
        torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6], dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(3, -1)
        .contiguous()
    )
    b_ready_cache_len = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    b_seq_len = torch.tensor([5, 7], dtype=torch.int32, device="cuda")
    b_start_loc = torch.tensor([0, 5], dtype=torch.int32, device="cuda")
    get_mrope_position_triton(
        b_image_start_idx,
        b_image_pos_delta,
        b_image_nums,
        b_image_start_num,
        b_image_len,
        b_image_cu_len,
        b_image_position_id,
        position_ids,
        b_ready_cache_len,
        b_seq_len,
        b_start_loc,
    )
    print(position_ids)
    """
    tensor([[ 0,  1,  2,  3,  4,  0,  0,  0,  0,  0,  1,  2],
        [ 0,  1,  2,  3,  4, 21, 21, 31, 31,  0,  1,  2],
        [ 0,  1,  2,  3,  4, 22, 22, 32, 32,  0,  1,  2]], device='cuda:0',
       dtype=torch.int32)
    """
