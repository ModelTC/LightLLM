import torch
import triton
import triton.language as tl


@triton.jit
def _get_mrope_position_triton(
    b_image_start_idx: torch.Tensor,
    b_image_thwd: torch.Tensor,
    b_image_thwd_stride0: torch.Tensor,
    b_image_nums: torch.Tensor,
    b_image_start_num: torch.Tensor,
    b_image_len: torch.Tensor,
    position_ids: torch.Tensor,
    position_ids_stride0: torch.Tensor,
    b_ready_cache_len: torch.Tensor,
    b_q_seq_len: torch.Tensor,
    b_start_loc: torch.Tensor,
    BLOCK_SIZE: tl.constexpr,
) -> torch.Tensor:
    cur_batch = tl.program_id(0)
    cache_len = tl.load(b_ready_cache_len + cur_batch)
    q_seq_len = tl.load(b_q_seq_len + cur_batch)
    image_num = tl.load(b_image_nums + cur_batch)
    image_start_num = tl.load(b_image_start_num + cur_batch)
    start_loc = tl.load(b_start_loc + cur_batch)
    for i in range(image_num):
        local_image_start_idx = tl.load(b_image_start_idx + image_start_num + i)
        image_start_idx = start_loc + local_image_start_idx - cache_len
        image_len = tl.load(b_image_len + image_start_num + i)
        image_h = tl.load(b_image_thwd + (image_start_num + i) * b_image_thwd_stride0 + 1)
        image_w = tl.load(b_image_thwd + (image_start_num + i) * b_image_thwd_stride0 + 2)
        for j in range(0, image_len, BLOCK_SIZE):
            off = j + tl.arange(0, BLOCK_SIZE)
            # 目前没考虑视频，所以t 恒为 0
            t_pos = local_image_start_idx + off * 0
            h_pos = local_image_start_idx + off // image_h
            w_pos = local_image_start_idx + off % image_w
            tl.store(
                position_ids + off + image_start_idx,
                t_pos,
                mask=(off < image_len)
                & (off + local_image_start_idx - cache_len < q_seq_len)
                & (local_image_start_idx - cache_len + off >= 0),
            )
            tl.store(
                position_ids + position_ids_stride0 + off + image_start_idx,
                h_pos,
                mask=(off < image_len)
                & (off + local_image_start_idx - cache_len < q_seq_len)
                & (local_image_start_idx - cache_len + off >= 0),
            )
            tl.store(
                position_ids + position_ids_stride0 * 2 + off + image_start_idx,
                w_pos,
                mask=(off < image_len)
                & (off + local_image_start_idx - cache_len < q_seq_len)
                & (local_image_start_idx - cache_len + off >= 0),
            )

    for i in range(image_num):
        local_image_start_idx = tl.load(b_image_start_idx + image_start_num + i)
        image_len = tl.load(b_image_len + image_start_num + i)
        image_delta = tl.load(b_image_thwd + (image_start_num + i) * b_image_thwd_stride0 + 3)
        image_end = local_image_start_idx + image_len - cache_len
        text_start = tl.maximum(0, image_end)
        for j in range(text_start, q_seq_len, BLOCK_SIZE):
            off = j + tl.arange(0, BLOCK_SIZE)
            t_pos = tl.load(position_ids + off + start_loc, mask=(off < q_seq_len), other=0.0) + image_delta
            h_pos = (
                tl.load(position_ids + position_ids_stride0 + off + start_loc, mask=(off < q_seq_len), other=0.0)
                + image_delta
            )
            w_pos = (
                tl.load(position_ids + position_ids_stride0 * 2 + off + start_loc, mask=(off < q_seq_len), other=0.0)
                + image_delta
            )
            tl.store(position_ids + off + start_loc, t_pos, mask=(off < q_seq_len))
            tl.store(position_ids + position_ids_stride0 + off + start_loc, h_pos, mask=(off < q_seq_len))
            tl.store(position_ids + position_ids_stride0 * 2 + off + start_loc, w_pos, mask=(off < q_seq_len))
    return


def get_mrope_position_triton(
    b_image_start_idx: torch.Tensor,
    b_image_thwd: torch.Tensor,
    b_image_nums: torch.Tensor,
    b_image_start_num: torch.Tensor,
    b_image_len: torch.Tensor,
    position_ids: torch.Tensor,
    b_ready_cache_len: torch.Tensor,
    b_q_seq_len: torch.Tensor,
    b_start_loc: torch.Tensor,
) -> torch.Tensor:

    batch_size = b_q_seq_len.shape[0]
    assert batch_size == b_image_nums.shape[0]
    grid = (batch_size,)
    BLOCK_SIZE = 64
    _get_mrope_position_triton[grid](
        b_image_start_idx=b_image_start_idx,
        b_image_thwd=b_image_thwd,
        b_image_thwd_stride0=b_image_thwd.stride(0),
        b_image_nums=b_image_nums,
        b_image_start_num=b_image_start_num,
        b_image_len=b_image_len,
        position_ids=position_ids,
        position_ids_stride0=position_ids.stride(0),
        b_ready_cache_len=b_ready_cache_len,
        b_q_seq_len=b_q_seq_len,
        b_start_loc=b_start_loc,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def test():
    b_image_start_idx = torch.tensor([0, 0, 4], dtype=torch.int32, device="cuda")
    b_image_thwd = torch.tensor([[1, 2, 2, -2], [1, 2, 2, -2], [1, 2, 2, -2]], dtype=torch.int32, device="cuda")
    b_image_nums = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    b_image_start_num = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    b_image_len = torch.tensor([4, 4, 4], dtype=torch.int32, device="cuda")
    position_ids = (
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(3, -1)
        .contiguous()
    )
    b_ready_cache_len = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    b_q_seq_len = torch.tensor([7, 13], dtype=torch.int32, device="cuda")
    b_start_loc = torch.tensor([0, 7], dtype=torch.int32, device="cuda")
    get_mrope_position_triton(
        b_image_start_idx,
        b_image_thwd,
        b_image_nums,
        b_image_start_num,
        b_image_len,
        position_ids,
        b_ready_cache_len,
        b_q_seq_len,
        b_start_loc,
    )

    # print(position_ids)
    old_value = torch.cat([position_ids[:, 2:7], position_ids[:, 7 + 2 :]], dim=1)

    position_ids = (
        torch.tensor([2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(3, -1)
        .contiguous()
    )
    b_ready_cache_len = torch.tensor([2, 2], dtype=torch.int32, device="cuda")
    b_q_seq_len = torch.tensor([5, 11], dtype=torch.int32, device="cuda")
    b_start_loc = torch.tensor([0, 5], dtype=torch.int32, device="cuda")

    get_mrope_position_triton(
        b_image_start_idx,
        b_image_thwd,
        b_image_nums,
        b_image_start_num,
        b_image_len,
        position_ids,
        b_ready_cache_len,
        b_q_seq_len,
        b_start_loc,
    )

    assert torch.equal(old_value, position_ids)

    """
    tensor([[0, 0, 0, 0, 2, 3, 4, 0, 0, 0, 0, 2, 2, 2, 2, 4, 5, 6, 7, 8],
        [0, 0, 1, 1, 2, 3, 4, 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8],
        [0, 1, 0, 1, 2, 3, 4, 0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8]],
       device='cuda:0', dtype=torch.int32)
    """
