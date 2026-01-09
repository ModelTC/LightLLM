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
    fn2 = lambda: repack_kv_index(req_to_token_indexs, b_req_idx, b_seq_len, b_start_loc, MAX_SEQ_LEN, output)
    ms1 = triton.testing.do_bench(fn1)
    ms2 = triton.testing.do_bench_cudagraph(fn2)
    print(ms1, ms2)
    assert torch.allclose(output.float(), ref.float())
