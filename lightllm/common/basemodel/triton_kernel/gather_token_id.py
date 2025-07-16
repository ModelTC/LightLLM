import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_gather_and_scatter(
    probs_idx,
    probs_sort,
    req_to_next_token_ids,
    req_to_next_token_probs,
    sampled_index,
    b_req_idx,
    probs_idx_stride,
    probs_sort_stride,
    req_to_next_token_ids_stride,
    req_to_next_token_probs_stride,
):
    cur_index = tl.program_id(0)
    cur_req_idx = tl.load(b_req_idx + cur_index)
    cur_sampled_index = tl.load(sampled_index + cur_index)
    cur_token_index = tl.load(probs_idx + cur_index * probs_idx_stride + cur_sampled_index)
    cur_token_probs = tl.load(probs_sort + cur_index * probs_sort_stride + cur_sampled_index)
    tl.store(req_to_next_token_ids + cur_req_idx * req_to_next_token_ids_stride, cur_token_index)
    tl.store(req_to_next_token_probs + cur_req_idx * req_to_next_token_probs_stride, tl.log(cur_token_probs))
    return


@torch.no_grad()
def gather_and_scatter_token_to_cpu(
    probs_idx: torch.Tensor,
    probs_sort: torch.Tensor,
    req_to_next_token_ids: torch.Tensor,
    req_to_next_token_probs: torch.Tensor,
    sampled_index: torch.Tensor,
    b_req_idx: torch.Tensor,
):
    """
    This function is used to gather the next_token_id(GPU tensor) and next_token_probs(GPU tensor)
    info to the req_to_next_token_ids and req_to_next_token_probs(CPU tensor).
    Args:
        probs_idx: (batch_size, vocab_size)
        probs_sort: (batch_size, vocab_size)
        req_to_next_token_ids: (max_req_num,)
        req_to_next_token_probs: (max_req_num,)
        sampled_index: (batch_size,)
        b_req_idx: (batch_size,)
    """
    assert probs_idx.shape == probs_sort.shape
    assert sampled_index.shape[0] == b_req_idx.shape[0]
    batch_size = b_req_idx.shape[0]
    grid = (batch_size,)
    num_warps = 1

    _fwd_kernel_gather_and_scatter[grid](
        probs_idx,
        probs_sort,
        req_to_next_token_ids,
        req_to_next_token_probs,
        sampled_index,
        b_req_idx,
        probs_idx.stride(0),
        probs_sort.stride(0),
        req_to_next_token_ids.stride(0),
        req_to_next_token_probs.stride(0),
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _fwd_kernel_scatter(
    token_info,
    req_to_token_info,
    b_req_idx,
    req_to_token_info_stride,
):
    cur_index = tl.program_id(0)
    cur_req_idx = tl.load(b_req_idx + cur_index)
    cur_token_info = tl.load(token_info + cur_index)
    tl.store(req_to_token_info + cur_req_idx * req_to_token_info_stride, cur_token_info)
    return


@torch.no_grad()
def scatter_token(token_info: torch.Tensor, req_to_token_info: torch.Tensor, b_req_idx: torch.Tensor):
    """
    This function is used to scatter the token_info(GPU tensor) to the req_to_token_info(CPU tensor).
    Args:
        token_info: (batch_size, vocab_size)
        req_to_token_info: (max_req_num,)
        b_req_idx: (batch_size,)
    """
    assert token_info.shape[0] == b_req_idx.shape[0]
    batch_size = b_req_idx.shape[0]
    grid = (batch_size,)
    num_warps = 1

    _fwd_kernel_scatter[grid](
        token_info,
        req_to_token_info,
        b_req_idx,
        req_to_token_info.stride(0),
        num_warps=num_warps,
        num_stages=1,
    )
    return


@triton.jit
def _fwd_kernel_gather(
    req_to_token_info,
    req_to_token_info_stride,
    output,
    b_req_idx,
):
    cur_index = tl.program_id(0)
    cur_req_idx = tl.load(b_req_idx + cur_index)
    cur_token_info = tl.load(req_to_token_info + cur_req_idx * req_to_token_info_stride)
    tl.store(output + cur_index, cur_token_info)
    return


def gather_token(req_to_token_info: torch.Tensor, b_req_idx: torch.Tensor):
    """
    This function is used to gather the token_info(CPU tensor) to the token_info(GPU tensor).
    Args:
        req_to_token_info: (max_req_num, max_mtp_step)
        b_req_idx: (batch_size,)
    Returns:
        output: (batch_size,)
    """
    batch_size = b_req_idx.shape[0]
    output = torch.empty_like(b_req_idx)
    grid = (batch_size,)
    num_warps = 1
    _fwd_kernel_gather[grid](
        req_to_token_info,
        req_to_token_info.stride(0),
        output,
        b_req_idx,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def _top_p_top_k(probs: torch.Tensor, top_ps: torch.Tensor, top_ks: torch.Tensor):
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)

    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    probs_sort[torch.arange(0, probs.shape[-1], device="cuda").view(1, -1) >= top_ks.view(-1, 1)] = 0.0

    return probs_sort, probs_idx


def test_gather_and_scatter_token_to_cpu():
    batch_size = 30
    vocab_size = 60000
    req_to_next_token_ids = torch.ones((1000,), dtype=torch.int32, pin_memory=True)
    req_to_next_token_probs = torch.ones((1000,), dtype=torch.float32, pin_memory=True)
    req_ids = torch.arange(20, 20 + batch_size, dtype=torch.int32).cuda()
    probs = torch.randn((batch_size, vocab_size)).cuda()
    top_ps = torch.rand((batch_size,)).cuda()
    top_ks = torch.ones((batch_size,), dtype=torch.int32).cuda()
    probs_sort, probs_idx = _top_p_top_k(probs, top_ps, top_ks)
    sampled_index = torch.multinomial(probs_sort, num_samples=1, replacement=True)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index)
    batch_next_token_probs = torch.gather(probs_sort, dim=1, index=sampled_index)

    gather_and_scatter_token_to_cpu(
        probs_idx, probs_sort, req_to_next_token_ids, req_to_next_token_probs, sampled_index, req_ids
    )
    diff_ids = (req_to_next_token_ids[20 : 20 + batch_size].cuda() - batch_next_token_ids.view(-1)).abs().max()
    diff_probs = (req_to_next_token_probs[20 : 20 + batch_size].cuda() - batch_next_token_probs.view(-1)).abs().max()
    assert diff_ids < 1e-6
    assert diff_probs < 1e-6
    print("test_gather_and_scatter_token_to_cpu passed")


def test_scatter_token_to_cpu():
    batch_size = 30
    req_to_token_info = torch.zeros((1000,), dtype=torch.float32, pin_memory=True)
    token_info = torch.randn((batch_size,)).cuda()
    req_ids = torch.arange(20, 20 + batch_size, dtype=torch.int32).cuda()
    scatter_token(token_info, req_to_token_info, req_ids)
    diff = (req_to_token_info[20 : 20 + batch_size].cuda() - token_info).abs().max()
    assert diff < 1e-6
    print("test_scatter_token_to_cpu passed")


def test_gather_token():
    batch_size = 30
    req_to_token_info = torch.zeros((1000,), dtype=torch.int32, pin_memory=True)
    token_info = torch.randn((batch_size,)).cuda()
    req_ids = torch.arange(20, 20 + batch_size, dtype=torch.int32).cuda()
    scatter_token(token_info, req_to_token_info, req_ids)
    output = gather_token(req_to_token_info, req_ids)
    diff = (token_info - output).abs().max()
    assert diff < 1e-6
    print("test_gather_token passed")


if __name__ == "__main__":
    test_gather_and_scatter_token_to_cpu()
    test_scatter_token_to_cpu()
    test_gather_token()
