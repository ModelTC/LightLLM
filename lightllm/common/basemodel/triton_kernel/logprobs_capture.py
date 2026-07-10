import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_prompt_logprobs_to_cpu(
    mem_indexes,
    top_token_ids,
    top_logprobs,
    top_token_ids_buffer_ptr,
    top_logprobs_buffer_ptr,
    token_count,
    kv_cache_size,
    TOPK: tl.constexpr,
    MAX_TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows = offsets // TOPK
    columns = offsets - rows * TOPK
    mask = rows < token_count
    mem_index = tl.load(mem_indexes + rows, mask=mask, other=-1).to(tl.int64)
    write_mask = mask & (mem_index >= 0) & (mem_index < kv_cache_size)
    source_offset = rows * TOPK + columns
    destination_offset = mem_index * MAX_TOPK + columns

    token_ids = tl.load(top_token_ids + source_offset, mask=write_mask, other=-1)
    token_ids_dst = tl.load(top_token_ids_buffer_ptr).to(tl.pointer_type(tl.int32))
    tl.store(token_ids_dst + destination_offset, token_ids, mask=write_mask)

    logprobs = tl.load(top_logprobs + source_offset, mask=write_mask, other=0.0)
    logprobs_dst = tl.load(top_logprobs_buffer_ptr).to(tl.pointer_type(tl.float32))
    tl.store(logprobs_dst + destination_offset, logprobs, mask=write_mask)


def scatter_prompt_logprobs_to_cpu(
    mem_indexes: torch.Tensor,
    top_token_ids: torch.Tensor,
    top_logprobs: torch.Tensor,
    top_token_ids_buffer_ptr: torch.Tensor,
    top_logprobs_buffer_ptr: torch.Tensor,
    kv_cache_size: int,
    max_topk: int,
) -> None:
    token_count, topk = top_token_ids.shape
    assert mem_indexes.is_cuda and mem_indexes.is_contiguous()
    assert mem_indexes.dtype in (torch.int32, torch.int64)
    assert mem_indexes.numel() == token_count
    assert top_token_ids.is_cuda and top_token_ids.is_contiguous()
    assert top_token_ids.dtype == torch.int32
    assert top_logprobs.is_cuda and top_logprobs.is_contiguous()
    assert top_logprobs.dtype == torch.float32
    assert top_logprobs.shape == top_token_ids.shape
    assert 0 < topk <= max_topk

    block = 1024
    _scatter_prompt_logprobs_to_cpu[(triton.cdiv(token_count * topk, block),)](
        mem_indexes=mem_indexes,
        top_token_ids=top_token_ids,
        top_logprobs=top_logprobs,
        top_token_ids_buffer_ptr=top_token_ids_buffer_ptr,
        top_logprobs_buffer_ptr=top_logprobs_buffer_ptr,
        token_count=token_count,
        kv_cache_size=kv_cache_size,
        TOPK=topk,
        MAX_TOPK=max_topk,
        BLOCK=block,
    )
