from typing import List, Optional, Tuple, Type
import torch
import triton
import triton.language as tl


def ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def moe_align_block_size_stage1(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    start_idx = pid * tokens_per_thread

    off_c = (pid + 1) * num_experts

    for i in range(tokens_per_thread):
        if start_idx + i < numel:
            idx = tl.load(topk_ids_ptr + start_idx + i)
            token_cnt = tl.load(tokens_cnts_ptr + off_c + idx)
            tl.store(tokens_cnts_ptr + off_c + idx, token_cnt + 1)


@triton.jit
def moe_align_block_size_stage2(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    last_cnt = 0
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last_cnt = last_cnt + token_cnt
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last_cnt)


@triton.jit
def moe_align_block_size_stage3(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    last_cumsum = 0
    off_cnt = num_experts * num_experts
    for i in range(1, num_experts + 1):
        token_cnt = tl.load(tokens_cnts_ptr + off_cnt + i - 1)
        last_cumsum = last_cumsum + tl.cdiv(token_cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last_cumsum)
    tl.store(total_tokens_post_pad_ptr, last_cumsum)


@triton.jit
def moe_align_block_size_stage4(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    total_tokens_post_pad_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = tl.load(cumsum_ptr + pid)
    end_idx = tl.load(cumsum_ptr + pid + 1)

    for i in range(start_idx, end_idx, block_size):
        tl.store(expert_ids_ptr + i // block_size, pid)

    start_idx = pid * tokens_per_thread
    off_t = pid * num_experts

    for i in range(start_idx, tl.minimum(start_idx + tokens_per_thread, numel)):
        expert_id = tl.load(topk_ids_ptr + i)
        token_cnt = tl.load(tokens_cnts_ptr + off_t + expert_id)
        rank_post_pad = token_cnt + tl.load(cumsum_ptr + expert_id)
        tl.store(sorted_token_ids_ptr + rank_post_pad, i)
        tl.store(tokens_cnts_ptr + off_t + expert_id, token_cnt + 1)


@torch.no_grad()
def moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    numel = topk_ids.numel()
    grid = (num_experts,)
    tokens_cnts = torch.zeros((num_experts + 1, num_experts), dtype=torch.int32, device="cuda")
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device="cuda")
    tokens_per_thread = ceil_div(numel, num_experts)

    moe_align_block_size_stage1[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
        BLOCK_SIZE=num_experts,
    )
    moe_align_block_size_stage2[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
        BLOCK_SIZE=num_experts,
    )
    moe_align_block_size_stage3[(1,)](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
        BLOCK_SIZE=num_experts,
    )
    moe_align_block_size_stage4[grid](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
        BLOCK_SIZE=num_experts,
    )


@torch.no_grad()
def torch_moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:

    tokens_cnts = topk_ids.new_zeros((topk_ids.shape[0], num_experts))
    tokens_cnts.scatter_(1, topk_ids, 1)
    tokens_cnts = tokens_cnts.sum(dim=0)
    cumsum = topk_ids.new_zeros(num_experts + 1)
    for i in range(num_experts):
        cumsum[i + 1] = cumsum[i] + ceil_div(tokens_cnts[i], block_size) * block_size
    num_tokens_post_pad[0] = cumsum[-1]
    expert_index = 0
    for i in range(0, num_tokens_post_pad[0], block_size):
        while i >= cumsum[expert_index + 1]:
            expert_index += 1
        experts_ids[i // block_size] = expert_index
    numel = topk_ids.numel()
    topk_ids = topk_ids.view(-1)
    tokens_cnts = torch.zeros_like(tokens_cnts)
    for i in range(numel):
        expert_id = topk_ids[i]
        rank_post_pad = tokens_cnts[expert_id] + cumsum[expert_id]
        sorted_token_ids[rank_post_pad] = i
        tokens_cnts[expert_id] += 1


def test():
    def generate_unique_rows_tensor(rows=59, cols=6, max_val=63):
        assert cols <= max_val + 1, "Number of columns cannot be greater than max_val + 1"

        tensor = torch.empty((rows, cols), dtype=torch.int64, device="cuda")

        for i in range(rows):
            row = torch.randperm(max_val + 1, dtype=torch.int64, device="cuda")[:cols]
            tensor[i] = row

        return tensor

    num_experts = 64
    topk_ids = generate_unique_rows_tensor(8192, 6, num_experts - 1)
    block_size = 16
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    sorted_ids_1 = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    sorted_ids_1.fill_(topk_ids.numel())
    expert_ids_1 = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad_1 = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    import time

    start_time = time.time()
    torch_moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_tokens_post_pad)
    end_time = time.time()
    print("torch cost: ", end_time - start_time)
    moe_align_block_size(topk_ids, num_experts, block_size, sorted_ids_1, expert_ids_1, num_tokens_post_pad_1)
    end_time1 = time.time()
    print("triton cost: ", end_time1 - end_time)
    assert torch.equal(sorted_ids, sorted_ids_1)
    assert torch.equal(expert_ids, expert_ids_1)
    assert torch.equal(num_tokens_post_pad, num_tokens_post_pad_1)


if __name__ == "__main__":
    test()
