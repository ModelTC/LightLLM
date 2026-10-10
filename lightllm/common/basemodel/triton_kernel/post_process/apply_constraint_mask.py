from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _apply_constraint_mask(
    logits,
    b_req_idx,
    b_mtp_index,
    req_to_bitmask,
    req_to_bitmask_enabled,
    logits_stride,
    mask_req_stride,
    mask_mtp_stride,
    VOCAB_SIZE: tl.constexpr,
    LOGITS_WIDTH: tl.constexpr,
    HAS_MTP_INDEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    prediction = tl.program_id(0)
    req_idx = tl.load(b_req_idx + prediction)
    token_ids = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    logits_ptr = logits + prediction * logits_stride + token_ids

    if tl.load(req_to_bitmask_enabled + req_idx):
        mtp_index = tl.load(b_mtp_index + prediction) if HAS_MTP_INDEX else 0
        mask_ptr = req_to_bitmask + req_idx * mask_req_stride + mtp_index * mask_mtp_stride
        word_ids = tl.program_id(1) * (BLOCK_SIZE // 32) + tl.arange(0, BLOCK_SIZE // 32)
        # Read each packed word once from host memory, then expand its 32 bits
        # in registers. Per-token loads waste mapped-memory transactions.
        mask_words = tl.load(mask_ptr + word_ids, mask=word_ids < tl.cdiv(VOCAB_SIZE, 32), other=0)
        allowed = ((mask_words[:, None] >> tl.arange(0, 32)[None, :]) & 1).reshape(BLOCK_SIZE)
        # Only forbidden logits need a write; allowed scores stay untouched.
        tl.store(
            logits_ptr,
            float("-inf"),
            mask=(token_ids < LOGITS_WIDTH) & ((allowed == 0) | (token_ids >= VOCAB_SIZE)),
        )
    elif LOGITS_WIDTH > VOCAB_SIZE:
        tl.store(logits_ptr, float("-inf"), mask=(token_ids >= VOCAB_SIZE) & (token_ids < LOGITS_WIDTH))


def apply_constraint_mask(
    logits: torch.Tensor,
    b_req_idx: torch.Tensor,
    req_to_bitmask: torch.Tensor,
    req_to_bitmask_enabled: torch.Tensor,
    vocab_size: int,
    b_mtp_index: Optional[torch.Tensor] = None,
) -> None:
    """Apply request-owned masks, reading the CPU-filled pinned buffers directly.

    b_mtp_index is the existing compacted verify position; ordinary decode and
    prefill use position zero. Call after penalties on the sampling stream.
    The existing post_handle completion event protects buffer reuse next step.
    """
    assert logits.is_cuda and logits.stride(1) == 1
    assert req_to_bitmask.is_pinned() and req_to_bitmask_enabled.is_pinned()
    if logits.shape[0] == 0:
        return
    block_size = 4096
    _apply_constraint_mask[(logits.shape[0], triton.cdiv(logits.shape[1], block_size))](
        logits,
        b_req_idx,
        b_mtp_index,
        req_to_bitmask,
        req_to_bitmask_enabled,
        logits.stride(0),
        req_to_bitmask.stride(0),
        req_to_bitmask.stride(1),
        VOCAB_SIZE=vocab_size,
        LOGITS_WIDTH=logits.shape[1],
        HAS_MTP_INDEX=b_mtp_index is not None,
        BLOCK_SIZE=block_size,
    )
