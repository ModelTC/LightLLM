import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_repack_kv_index(
    req_to_token_indexs,
    b_req_idx,
    out_page_indices,
    b_token_len,
    b_page_start_loc,
    req_to_token_stride_0,
    PAGE_SIZE: tl.constexpr,
    PAGE_BLOCK: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    page_block_idx = tl.program_id(1)

    # token 长度向上取整为页数，后续输出偏移以页为单位。
    cur_page_len = tl.cdiv(tl.load(b_token_len + cur_batch), PAGE_SIZE)
    cur_batch_req_idx = tl.load(b_req_idx + cur_batch)
    cur_page_start = tl.load(b_page_start_loc + cur_batch)

    page_offsets = page_block_idx * PAGE_BLOCK + tl.arange(0, PAGE_BLOCK)
    block_end_page = tl.minimum((page_block_idx + 1) * PAGE_BLOCK, cur_page_len)
    # 每个逻辑页只读取页首 token 的物理索引。
    physical_token_index = tl.load(
        req_to_token_indexs + req_to_token_stride_0 * cur_batch_req_idx + page_offsets * PAGE_SIZE,
        mask=page_offsets < block_end_page,
        other=0,
    )
    # 物理 token 索引除以页大小，转换为 FlashInfer 所需的物理页号。
    out_page_index_ptr = out_page_indices + cur_page_start + page_offsets
    tl.store(out_page_index_ptr, physical_token_index // PAGE_SIZE, mask=page_offsets < block_end_page)
    return


@torch.no_grad()
def repack_kv_index(
    req_to_token_indexs,  # [请求槽位数, token 容量]，存放各请求的物理 token 索引
    b_req_idx,  # [batch_size]，当前 batch 对应的请求槽位编号
    b_token_len,  # [batch_size]，每个请求的有效 KV token 数
    b_page_start_loc,  # [batch_size]，每个请求在输出页索引数组中的起始位置，单位为页
    max_token_len,  # 当前 batch 的最大 KV token 长度，用于确定 kernel 的处理范围
    out_page_indices,  # 一维输出数组，按 batch 顺序紧凑存放各请求的物理页号，原地写入
    page_size=1,  # 每个 KV 页包含的 token 数
):
    batch_size = b_req_idx.shape[0]
    PAGE_BLOCK = 64  # 每个 program 处理的页数
    grid = (
        batch_size,
        triton.cdiv(max_token_len, page_size * PAGE_BLOCK),
    )

    _fwd_kernel_repack_kv_index[grid](
        req_to_token_indexs,
        b_req_idx,
        out_page_indices,
        b_token_len,
        b_page_start_loc,
        req_to_token_indexs.stride(0),
        PAGE_SIZE=page_size,
        PAGE_BLOCK=PAGE_BLOCK,
        num_warps=8,
        num_stages=1,
    )
    return


def repack_kv_ref(req_to_token_indexs, b_req_idx, b_seq_len, b_start_loc, output):
    for b, sl, start in zip(b_req_idx, b_seq_len, b_start_loc):
        output[start : start + sl] = req_to_token_indexs[b][:sl]


if __name__ == "__main__":
    import torch.nn.functional as F

    BATCH, MAX_SEQ_LEN = 10, 1024
    rand_idx = torch.randperm(2 * MAX_SEQ_LEN * BATCH).cuda().int()
    b_req_idx = torch.randperm(BATCH).cuda().int()
    b_seq_len = torch.randint(1, MAX_SEQ_LEN, (BATCH,)).cuda().int()
    req_to_token_indexs = torch.zeros((2 * BATCH, 2 * MAX_SEQ_LEN)).cuda().int()
    b_start_loc = (
        torch.cat([torch.zeros([1], device=b_seq_len.device, dtype=b_seq_len.dtype), b_seq_len[0:-1].cumsum(0)])
        .cuda()
        .int()
    )

    output = torch.zeros((b_seq_len.sum(),)).cuda().int()
    ref = torch.zeros((b_seq_len.sum(),)).cuda().int()
    for b, sl, start in zip(b_req_idx, b_seq_len, b_start_loc):
        req_to_token_indexs[b][:sl] = rand_idx[start : start + sl]

    fn1 = lambda: repack_kv_ref(req_to_token_indexs, b_req_idx, b_seq_len, b_start_loc, ref)
    fn2 = lambda: repack_kv_index(
        req_to_token_indexs=req_to_token_indexs,
        b_req_idx=b_req_idx,
        b_token_len=b_seq_len,
        b_page_start_loc=b_start_loc,
        max_token_len=MAX_SEQ_LEN,
        out_page_indices=output,
    )
    ms1 = triton.testing.do_bench(fn1)
    ms2 = triton.testing.do_bench_cudagraph(fn2)
    print(ms1, ms2)
    assert torch.allclose(output.float(), ref.float())
