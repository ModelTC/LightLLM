import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_scatter(
    next_token_ids,
    req_to_next_token_ids,
    b_req_idx,
    b_mtp_index,
    b_has_out,
    req_to_next_token_ids_stride,
    req_to_next_token_ids_stride_1,
    num_size,
    HAS_OUT_IS_NONE: tl.constexpr,
    BLOCK: tl.constexpr,
    OLD_VERSION_TRITON: tl.constexpr,
):
    block_index = tl.program_id(0)
    block_range = block_index * BLOCK + tl.arange(0, BLOCK)
    block_mask = block_range < num_size

    cur_req_idx = tl.load(b_req_idx + block_range, mask=block_mask)
    cur_mtp_index = tl.load(b_mtp_index + block_range, mask=block_mask)
    cur_next_token_id = tl.load(next_token_ids + block_range, mask=block_mask)

    if not HAS_OUT_IS_NONE:
        cur_has_out = tl.load(b_has_out + block_range, mask=block_mask, other=False)
        if OLD_VERSION_TRITON:
            cur_has_out = cur_has_out != 0
        tl.store(
            req_to_next_token_ids + cur_req_idx * req_to_next_token_ids_stride + cur_mtp_index,
            cur_next_token_id,
            mask=cur_has_out & block_mask,
        )
    else:
        tl.store(
            req_to_next_token_ids + cur_req_idx * req_to_next_token_ids_stride + cur_mtp_index,
            cur_next_token_id,
            mask=block_mask,
        )

    return


@torch.no_grad()
def scatter_token(
    next_token_ids: torch.Tensor,
    req_to_next_token_ids: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_mtp_index: torch.Tensor,
    b_has_out: torch.Tensor = None,
):
    """
    This function is used to scatter the token_info(GPU tensor) to the req_to_token_info(CPU tensor).
    Args:
        next_token_ids: (batch_size,)
        req_to_next_token_ids: (max_req_num, max_mtp_step)
        b_req_idx: (batch_size,)
        b_mtp_index: (batch_size,)
    """
    assert (
        next_token_ids.shape[0] == b_req_idx.shape[0]
    ), f"batch size not match, {next_token_ids.shape[0]} != {b_req_idx.shape[0]}"
    batch_size = b_req_idx.shape[0]
    BLOCK = 256

    grid = (triton.cdiv(batch_size, BLOCK),)
    num_warps = 1

    _fwd_kernel_scatter[grid](
        next_token_ids=next_token_ids,
        req_to_next_token_ids=req_to_next_token_ids,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        b_has_out=b_has_out,
        req_to_next_token_ids_stride=req_to_next_token_ids.stride(0),
        req_to_next_token_ids_stride_1=req_to_next_token_ids.stride(1),
        num_size=batch_size,
        HAS_OUT_IS_NONE=b_has_out is None,
        BLOCK=BLOCK,
        OLD_VERSION_TRITON=triton.__version__ < "3.2.0",
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _fwd_kernel_gather(
    req_to_next_token_ids,
    req_to_next_token_ids_stride,
    req_to_next_token_ids_stride_1,
    output,
    b_req_idx,
    b_mtp_index,
    num_size,
    BLOCK: tl.constexpr,
):
    block_index = tl.program_id(0)
    block_range = block_index * BLOCK + tl.arange(0, BLOCK)
    block_mask = block_range < num_size
    cur_req_idx = tl.load(b_req_idx + block_range, mask=block_mask)
    cur_mtp_index = tl.load(b_mtp_index + block_range, mask=block_mask)
    cur_next_token_id = tl.load(
        req_to_next_token_ids + cur_req_idx * req_to_next_token_ids_stride + cur_mtp_index, mask=block_mask
    )
    tl.store(output + block_range, cur_next_token_id, mask=block_mask)
    return


def gather_token(req_to_next_token_ids: torch.Tensor, b_req_idx: torch.Tensor, b_mtp_index: torch.Tensor):
    """
    This function is used to gather the token_info(CPU tensor) to the token_info(GPU tensor).
    Args:
        req_to_token_info: (max_req_num, max_mtp_step)
        b_req_idx: (batch_size,)
        b_mtp_index: (batch_size,)
    Returns:
        output: (batch_size,)
    """
    batch_size = b_req_idx.shape[0]
    output = torch.empty(batch_size, dtype=req_to_next_token_ids.dtype, device="cuda")
    BLOCK = 256
    grid = (triton.cdiv(batch_size, BLOCK),)
    num_warps = 1
    _fwd_kernel_gather[grid](
        req_to_next_token_ids=req_to_next_token_ids,
        req_to_next_token_ids_stride=req_to_next_token_ids.stride(0),
        req_to_next_token_ids_stride_1=req_to_next_token_ids.stride(1),
        output=output,
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        num_size=batch_size,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


@triton.jit
def _fwd_kernel_gather_prefill_decode_mixed(
    input_ids,
    req_to_next_token_ids,
    req_to_next_token_ids_stride,
    req_to_next_token_ids_stride_1,
    b_req_idx,
    b_mtp_index,
    b_is_decode_req,
    b_prefill_start_loc,
    num_size,
    BLOCK: tl.constexpr,
):
    block_index = tl.program_id(0)
    block_range = block_index * BLOCK + tl.arange(0, BLOCK)
    block_mask = block_range < num_size
    cur_req_idx = tl.load(b_req_idx + block_range, mask=block_mask)
    cur_mtp_index = tl.load(b_mtp_index + block_range, mask=block_mask)
    cur_next_token_id = tl.load(
        req_to_next_token_ids + cur_req_idx * req_to_next_token_ids_stride + cur_mtp_index, mask=block_mask
    )
    cur_is_decode_req = tl.load(b_is_decode_req + block_range, mask=block_mask, other=False)
    cur_prefill_start_loc = tl.load(b_prefill_start_loc + block_range, mask=block_mask, other=-1)

    tl.store(input_ids + cur_prefill_start_loc, cur_next_token_id, mask=block_mask & cur_is_decode_req)
    return


def gather_token_prefill_decode_mixed(
    input_ids: torch.Tensor,
    req_to_next_token_ids: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_mtp_index: torch.Tensor,
    b_is_decode_req: torch.Tensor,
    b_prefill_start_loc: torch.Tensor,
):
    """
    This function is used to gather the token_info(CPU tensor) to the token_info(GPU tensor).
    Args:
        input_ids: (batch_size,)
        req_to_next_token_ids: (max_req_num, max_mtp_step)
        b_req_idx: (batch_size,)
        b_mtp_index: (batch_size,)
        b_is_decode_req: (batch_size,)
        b_prefill_start_loc: (batch_size,)
    Returns:
        input_ids:
    """
    batch_size = b_req_idx.shape[0]
    BLOCK = 256
    grid = (triton.cdiv(batch_size, BLOCK),)
    num_warps = 1
    _fwd_kernel_gather_prefill_decode_mixed[grid](
        input_ids=input_ids,
        req_to_next_token_ids=req_to_next_token_ids,
        req_to_next_token_ids_stride=req_to_next_token_ids.stride(0),
        req_to_next_token_ids_stride_1=req_to_next_token_ids.stride(1),
        b_req_idx=b_req_idx,
        b_mtp_index=b_mtp_index,
        b_is_decode_req=b_is_decode_req,
        b_prefill_start_loc=b_prefill_start_loc,
        num_size=batch_size,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return input_ids


def test_scatter_token_to_cpu():
    batch_size = 30
    req_to_token_info = torch.zeros((1000, 1), dtype=torch.float32, pin_memory=True)
    token_info = torch.randn((batch_size,)).cuda()
    req_ids = torch.arange(20, 20 + batch_size, dtype=torch.int32).cuda()
    mtp_index = torch.zeros((batch_size,), dtype=torch.int32).cuda()
    scatter_token(token_info, req_to_token_info, req_ids, mtp_index)
    diff = (req_to_token_info[20 : 20 + batch_size].cuda().view(-1) - token_info).abs().max()
    assert diff < 1e-6
    print("test_scatter_token_to_cpu passed")


def test_gather_token():
    batch_size = 30
    req_to_token_info = torch.zeros((1000, 1), dtype=torch.float32, pin_memory=True)
    token_info = torch.randn((batch_size,)).cuda()
    req_ids = torch.arange(20, 20 + batch_size, dtype=torch.int32).cuda()
    mtp_index = torch.zeros((batch_size,), dtype=torch.int32).cuda()
    scatter_token(token_info, req_to_token_info, req_ids, mtp_index)
    output = gather_token(req_to_token_info, req_ids, mtp_index)
    diff = (token_info - output).abs().max()
    assert diff < 1e-6
    print("test_gather_token passed")


def _ref_gather_token_prefill_decode_mixed(
    input_ids: torch.Tensor,
    req_to_next_token_ids: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_mtp_index: torch.Tensor,
    b_is_decode_req: torch.Tensor,
    b_prefill_start_loc: torch.Tensor,
) -> torch.Tensor:
    out = input_ids.clone()
    table = req_to_next_token_ids.detach().cpu()
    req_idx_cpu = b_req_idx.detach().cpu()
    mtp_cpu = b_mtp_index.detach().cpu()
    is_decode_cpu = b_is_decode_req.detach().cpu()
    start_loc_cpu = b_prefill_start_loc.detach().cpu()
    for i in range(req_idx_cpu.shape[0]):
        if is_decode_cpu[i].item():
            rid = int(req_idx_cpu[i].item())
            mid = int(mtp_cpu[i].item())
            loc = int(start_loc_cpu[i].item())
            out[loc] = table[rid, mid]
    return out


def _run_gather_token_prefill_decode_mixed_case(
    input_ids: torch.Tensor,
    req_to_next_token_ids: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_mtp_index: torch.Tensor,
    b_is_decode_req: torch.Tensor,
    b_prefill_start_loc: torch.Tensor,
):
    input_cuda = input_ids.clone().cuda()
    req_table = req_to_next_token_ids.cuda()
    b_req_idx_cuda = b_req_idx.cuda()
    b_mtp_index_cuda = b_mtp_index.cuda()
    b_is_decode_cuda = b_is_decode_req.cuda()
    b_start_loc_cuda = b_prefill_start_loc.cuda()

    expected = _ref_gather_token_prefill_decode_mixed(
        input_cuda,
        req_table,
        b_req_idx_cuda,
        b_mtp_index_cuda,
        b_is_decode_cuda,
        b_start_loc_cuda,
    )
    gather_token_prefill_decode_mixed(
        input_cuda,
        req_table,
        b_req_idx_cuda,
        b_mtp_index_cuda,
        b_is_decode_cuda,
        b_start_loc_cuda,
    )
    diff = (input_cuda - expected).abs().max()
    assert diff < 1e-6, f"max diff {diff.item()}"


def test_gather_token_prefill_decode_mixed_decode_only():
    """仅 decode 行：按 b_prefill_start_loc 写入 req_to_next_token_ids 中的 next token。"""
    req_to_next_token_ids = torch.zeros((32, 4), dtype=torch.int64, device="cuda")
    req_to_next_token_ids[3, 0] = 42
    req_to_next_token_ids[7, 0] = 99
    req_to_next_token_ids[11, 2] = 17

    input_ids = torch.tensor([0, 0, 0], dtype=torch.int64, device="cuda")
    b_req_idx = torch.tensor([3, 7, 11], dtype=torch.int32, device="cuda")
    b_mtp_index = torch.tensor([0, 0, 2], dtype=torch.int32, device="cuda")
    b_is_decode_req = torch.tensor([True, True, True], dtype=torch.bool, device="cuda")
    b_prefill_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")

    _run_gather_token_prefill_decode_mixed_case(
        input_ids, req_to_next_token_ids, b_req_idx, b_mtp_index, b_is_decode_req, b_prefill_start_loc
    )
    print("test_gather_token_prefill_decode_mixed_decode_only passed")


def test_gather_token_prefill_decode_mixed_mixed_batch():
    """prefill + decode 混合：仅 decode 位置被覆盖，prefill token 保持不变。"""
    req_to_next_token_ids = torch.zeros((16, 2), dtype=torch.int64, device="cuda")
    req_to_next_token_ids[5, 0] = 9001

    # prefill [10,11,12] | decode placeholder | prefill [20,21]
    input_ids = torch.tensor([10, 11, 12, -1, 20, 21], dtype=torch.int64, device="cuda")
    b_req_idx = torch.tensor([0, 5, 1], dtype=torch.int32, device="cuda")
    b_mtp_index = torch.tensor([0, 0, 0], dtype=torch.int32, device="cuda")
    b_is_decode_req = torch.tensor([False, True, False], dtype=torch.bool, device="cuda")
    b_q_seq_len = torch.tensor([3, 1, 2], dtype=torch.int32, device="cuda")
    b_prefill_start_loc = b_q_seq_len.cumsum(dim=0, dtype=torch.int32) - b_q_seq_len

    _run_gather_token_prefill_decode_mixed_case(
        input_ids, req_to_next_token_ids, b_req_idx, b_mtp_index, b_is_decode_req, b_prefill_start_loc
    )
    print("test_gather_token_prefill_decode_mixed_mixed_batch passed")


def test_gather_token_prefill_decode_mixed_prefill_only_unchanged():
    """无 decode 行时 input_ids 不应被修改。"""
    req_to_next_token_ids = torch.full((8, 1), 777, dtype=torch.int64, device="cuda")
    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    b_req_idx = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    b_mtp_index = torch.zeros(3, dtype=torch.int32, device="cuda")
    b_is_decode_req = torch.zeros(3, dtype=torch.bool, device="cuda")
    b_q_seq_len = torch.tensor([2, 1, 1], dtype=torch.int32, device="cuda")
    b_prefill_start_loc = b_q_seq_len.cumsum(dim=0, dtype=torch.int32) - b_q_seq_len

    before = input_ids.clone()
    gather_token_prefill_decode_mixed(
        input_ids,
        req_to_next_token_ids,
        b_req_idx,
        b_mtp_index,
        b_is_decode_req,
        b_prefill_start_loc,
    )
    assert torch.equal(input_ids, before)
    print("test_gather_token_prefill_decode_mixed_prefill_only_unchanged passed")


def test_gather_token_prefill_decode_mixed_large_batch():
    """batch_size > 256，覆盖多 block 的 triton grid。"""
    batch_size = 300
    max_req = 400
    req_to_next_token_ids = torch.arange(max_req * 2, dtype=torch.int64, device="cuda").view(max_req, 2)
    input_ids = torch.zeros(batch_size, dtype=torch.int64, device="cuda")
    b_req_idx = torch.arange(10, 10 + batch_size, dtype=torch.int32, device="cuda")
    b_mtp_index = (b_req_idx % 2).to(torch.int32)
    b_is_decode_req = torch.ones(batch_size, dtype=torch.bool, device="cuda")
    b_prefill_start_loc = torch.arange(batch_size, dtype=torch.int32, device="cuda")

    _run_gather_token_prefill_decode_mixed_case(
        input_ids, req_to_next_token_ids, b_req_idx, b_mtp_index, b_is_decode_req, b_prefill_start_loc
    )
    print("test_gather_token_prefill_decode_mixed_large_batch passed")


def test_gather_token_prefill_decode_mixed_roundtrip_with_scatter():
    """scatter_token 写入后，mixed gather 能读回同一 next token。"""
    batch_size = 16
    req_to_next_token_ids = torch.zeros((64, 3), dtype=torch.float32, pin_memory=True)
    token_info = torch.arange(100, 100 + batch_size, dtype=torch.float32, device="cuda")
    b_req_idx = torch.arange(4, 4 + batch_size, dtype=torch.int32, device="cuda")
    b_mtp_index = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    scatter_token(token_info, req_to_next_token_ids, b_req_idx, b_mtp_index)

    input_ids = torch.zeros(batch_size, dtype=torch.int64, device="cuda")
    b_is_decode_req = torch.ones(batch_size, dtype=torch.bool, device="cuda")
    b_prefill_start_loc = torch.arange(batch_size, dtype=torch.int32, device="cuda")

    gather_token_prefill_decode_mixed(
        input_ids,
        req_to_next_token_ids,
        b_req_idx,
        b_mtp_index,
        b_is_decode_req,
        b_prefill_start_loc,
    )
    assert torch.equal(input_ids, token_info.to(torch.int64))
    print("test_gather_token_prefill_decode_mixed_roundtrip_with_scatter passed")


if __name__ == "__main__":
    test_scatter_token_to_cpu()
    test_gather_token()
    test_gather_token_prefill_decode_mixed_decode_only()
    test_gather_token_prefill_decode_mixed_mixed_batch()
    test_gather_token_prefill_decode_mixed_prefill_only_unchanged()
    test_gather_token_prefill_decode_mixed_large_batch()
    test_gather_token_prefill_decode_mixed_roundtrip_with_scatter()
