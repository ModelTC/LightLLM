import triton
import triton.language as tl


@triton.jit
def page_table_copy_kernel(
    page_table_ptr,
    req_to_token_indexs_ptr,
    b_req_idx,
    max_seq_len_k,
    b_req_idx_stride_0,
    page_table_stride_0,
    page_table_stride_1,
    req_to_token_stride_0,
    req_to_token_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    cur_batch = tl.program_id(axis=0)
    cur_block = tl.program_id(axis=1)
    cur_req_idx = tl.load(b_req_idx + cur_batch * b_req_idx_stride_0)

    offs = cur_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < max_seq_len_k

    input_pos = cur_req_idx * req_to_token_stride_0 + offs * req_to_token_stride_1
    output_pos = cur_batch * page_table_stride_0 + offs * page_table_stride_1

    mem_index = tl.load(req_to_token_indexs_ptr + input_pos, mask=mask)
    tl.store(page_table_ptr + output_pos, mem_index, mask=mask)


def page_table_copy(
    page_table,  # destination tensor [batch, seq]
    req_to_token_indexs,  # source tensor [batch, seq]
    b_req_idx,  # request index to copy from
    max_seq_len_k,  # sequence length to copy
):
    assert page_table.dim() == 2, "page_table should be 2D"
    assert req_to_token_indexs.dim() == 2, "req_to_token_indexs should be 2D"

    batch_size = page_table.size(0)
    BLOCK_SIZE = 128

    grid = (batch_size, triton.cdiv(max_seq_len_k, BLOCK_SIZE))

    page_table_copy_kernel[grid](
        page_table_ptr=page_table,
        req_to_token_indexs_ptr=req_to_token_indexs,
        b_req_idx=b_req_idx,
        max_seq_len_k=max_seq_len_k,
        b_req_idx_stride_0=b_req_idx.stride(0),
        page_table_stride_0=page_table.stride(0),
        page_table_stride_1=page_table.stride(1),
        req_to_token_stride_0=req_to_token_indexs.stride(0),
        req_to_token_stride_1=req_to_token_indexs.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )


import torch


def test_page_table_copy():
    batch_size, seq_len = 2, 8

    req_to_token_indexs = torch.arange(batch_size * seq_len, dtype=torch.int32).reshape(batch_size, seq_len).cuda()

    page_table = torch.full((batch_size, seq_len), -1, dtype=torch.int32, device="cuda")

    b_req_idx = torch.tensor([0, 2, 1, 3], dtype=torch.int32, device="cuda")[::2]
    print(b_req_idx.stride())
    max_seq_len_k = seq_len

    page_table_copy(page_table, req_to_token_indexs, b_req_idx, max_seq_len_k)

    print("req_to_token_indexs:")
    print(req_to_token_indexs.cpu().numpy())
    print("b_req_idx:", b_req_idx.cpu().numpy())
    print("page_table:")
    print(page_table.cpu().numpy())

    for batch in range(batch_size):
        src_idx = b_req_idx[batch].item()
        expected = req_to_token_indexs[src_idx].cpu().numpy()
        got = page_table[batch].cpu().numpy()
        assert (expected == got).all(), f"Batch {batch} mismatch: expected {expected}, got {got}"

    print("✅ Test passed!")


if __name__ == "__main__":
    test_page_table_copy()
