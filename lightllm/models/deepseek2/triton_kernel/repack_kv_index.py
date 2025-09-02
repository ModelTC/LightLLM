import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_repack_kv_index(
    kv_index,
    req_index,
    out_kv_index,
    seq_len,
    start_loc,
    kv_stride_h,
    SEQ_BLOCK: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    start_seq_n = tl.program_id(1)

    cur_batch_seq_len = tl.load(seq_len + cur_batch)
    cur_batch_req_idx = tl.load(req_index + cur_batch)
    cur_batch_start_loc = tl.load(start_loc + cur_batch)

    offs_seq = start_seq_n * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
    block_end_loc = tl.minimum((start_seq_n + 1) * SEQ_BLOCK, cur_batch_seq_len)
    kv_index_data = tl.load(
        kv_index + kv_stride_h * cur_batch_req_idx + offs_seq,
        mask=offs_seq < block_end_loc,
        other=0,
    )
    out_kv_index_ptr = out_kv_index + cur_batch_start_loc + offs_seq
    tl.store(out_kv_index_ptr, kv_index_data, mask=offs_seq < block_end_loc)
    return


@triton.jit
def _fwd_kernel_repack_page_kv_index_from_tokens(
    req_to_token_indexs,
    req_index,
    out_kv_index,
    seq_len,
    start_loc,
    page_size,
    token_stride_h,
    SEQ_BLOCK: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    start_seq_n = tl.program_id(1)

    cur_batch_seq_len = tl.load(seq_len + cur_batch)
    cur_batch_req_idx = tl.load(req_index + cur_batch)
    cur_batch_start_loc = tl.load(start_loc + cur_batch)

    offs_seq = (start_seq_n * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)) * page_size
    block_end_loc = (tl.minimum((start_seq_n + 1) * SEQ_BLOCK, cur_batch_seq_len)) * page_size
    token_data = tl.load(
        req_to_token_indexs + token_stride_h * cur_batch_req_idx + offs_seq,
        mask=offs_seq < block_end_loc,
        other=0,
    )
    valid_mask = (token_data % page_size) == 0
    valid_mask = valid_mask & (token_data > 0)  # 确保是有效的 token 索引
    page_data = tl.where(valid_mask, token_data // page_size, 0)

    offs_seq = start_seq_n * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
    block_end_loc = tl.minimum((start_seq_n + 1) * SEQ_BLOCK, cur_batch_seq_len)
    out_kv_index_ptr = out_kv_index + cur_batch_start_loc + offs_seq
    tl.store(out_kv_index_ptr, page_data, mask=offs_seq < block_end_loc)
    return


@torch.no_grad()
def repack_kv_index(kv_index, req_index, seq_len, start_loc, max_seq_len, out_kv_index):
    batch_size = req_index.shape[0]
    # flashinfer requires out_kv_index to be zeroed before use
    out_kv_index.zero_()
    BLOCK = 64
    grid = (
        batch_size,
        triton.cdiv(max_seq_len, BLOCK),
    )

    _fwd_kernel_repack_kv_index[grid](
        kv_index,
        req_index,
        out_kv_index,
        seq_len,
        start_loc,
        kv_index.stride(0),
        SEQ_BLOCK=BLOCK,
        num_warps=8,
        num_stages=1,
    )
    return


@torch.no_grad()
def repack_paged_kv_index_from_tokens(
    req_to_token_indexs, req_index, seq_len, start_loc, max_seq_len, page_size, out_kv_index
):
    batch_size = req_index.shape[0]
    out_kv_index.zero_()

    BLOCK = 64
    grid = (
        batch_size,
        triton.cdiv(max_seq_len, BLOCK),
    )

    _fwd_kernel_repack_page_kv_index_from_tokens[grid](
        req_to_token_indexs,
        req_index,
        out_kv_index,
        seq_len,
        start_loc,
        page_size,
        req_to_token_indexs.stride(0),
        SEQ_BLOCK=BLOCK,
        num_warps=8,
        num_stages=1,
    )
    return


def ref_repack_page_kv_index_with_token_input(
    req_to_token_indexs, req_index, seq_len, start_loc, max_seq_len, page_size, out_kv_index
):
    page_indexs = torch.zeros_like(req_to_token_indexs)
    valid_mask = req_to_token_indexs % page_size == 0
    batch_size, seq_len_dim = req_to_token_indexs.shape
    valid_positions = torch.cumsum(valid_mask.int(), dim=1) - 1
    batch_indices = torch.arange(batch_size, device=req_to_token_indexs.device).unsqueeze(1).expand(-1, seq_len_dim)
    page_indexs.view(-1).scatter_add_(
        0,
        (batch_indices * seq_len_dim + torch.where(valid_mask, valid_positions, 0)).flatten(),
        (torch.where(valid_mask, req_to_token_indexs // page_size, 0) * valid_mask.int()).flatten(),
    )

    repack_kv_index(page_indexs, req_index, seq_len, start_loc, max_seq_len, out_kv_index)


def repack_kv_ref(req_to_token_indexs, b_req_idx, b_seq_len, b_start_loc, output):
    for b, sl, start in zip(b_req_idx, b_seq_len, b_start_loc):
        output[start : start + sl] = req_to_token_indexs[b][:sl]


if __name__ == "__main__":
    import torch.nn.functional as F

    BATCH, MAX_SEQ_LEN = 10, 1024
    PAGE_SIZE = 64
    rand_idx = torch.randperm(2 * MAX_SEQ_LEN * BATCH).cuda().int()
    b_req_idx = torch.randperm(BATCH).cuda().int()
    b_seq_len = torch.randint(1, MAX_SEQ_LEN, (BATCH,)).cuda().int()
    req_to_token_indexs = torch.zeros((2 * BATCH, 2 * MAX_SEQ_LEN)).cuda().int()
    b_start_loc = (
        torch.cat([torch.zeros([1], device=b_seq_len.device, dtype=b_seq_len.dtype), b_seq_len[0:-1].cumsum(0)])
        .cuda()
        .int()
    )

    # 为每个batch生成基于page的连续索引
    for b in range(2 * BATCH):
        start_page_id = b * 100  # 确保不同batch有不同的page ID范围
        for token_idx in range(2 * MAX_SEQ_LEN):
            page_offset = token_idx // PAGE_SIZE
            token_in_page = token_idx % PAGE_SIZE
            page_id = start_page_id + page_offset
            token_index = page_id * PAGE_SIZE + token_in_page
            req_to_token_indexs[b, token_idx] = token_index

    output = torch.zeros((b_seq_len.sum(),)).cuda().int()
    ref = torch.zeros((b_seq_len.sum(),)).cuda().int()

    fn1 = lambda: repack_kv_ref(req_to_token_indexs, b_req_idx, b_seq_len, b_start_loc, ref)
    fn2 = lambda: repack_kv_index(req_to_token_indexs, b_req_idx, b_seq_len, b_start_loc, MAX_SEQ_LEN, output)
    ms1 = triton.testing.do_bench(fn1)
    ms2 = triton.testing.do_bench_cudagraph(fn2)
    print(f"repack_kv_index: ref={ms1:.3f}ms, triton={ms2:.3f}ms")
    assert torch.allclose(output.float(), ref.float())

    b_page_len = triton.cdiv(b_seq_len, PAGE_SIZE)
    page_output = torch.zeros((b_page_len.sum(),)).cuda().int()
    page_ref = torch.zeros((b_page_len.sum(),)).cuda().int()
    b_start_loc[1:] = b_page_len.cumsum(0)[:-1]
    max_seq_len = triton.cdiv(MAX_SEQ_LEN, PAGE_SIZE)
    fn3 = lambda: ref_repack_page_kv_index_with_token_input(
        req_to_token_indexs, b_req_idx, b_page_len, b_start_loc, max_seq_len, PAGE_SIZE, page_ref
    )
    fn4 = lambda: repack_paged_kv_index_from_tokens(
        req_to_token_indexs, b_req_idx, b_page_len, b_start_loc, max_seq_len, PAGE_SIZE, page_output
    )
    ms3 = triton.testing.do_bench(fn3)
    ms4 = triton.testing.do_bench_cudagraph(fn4)
    print(f"repack_paged_kv_index_from_tokens: ref={ms3:.3f}ms, triton={ms4:.3f}ms")
    assert torch.allclose(page_output.float(), page_ref.float())
