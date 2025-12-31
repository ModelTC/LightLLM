import torch
import triton
import triton.language as tl


@triton.jit
def alloc_buffer_for_req_kernel(
    req_index_ptr,  # [num_reqs] - indices of requests to allocate buffers for
    buffer_indexes_ptr,  # [num_reqs * num_buffers_per_req] - buffer indices to assign (from CPU)
    req_to_buffer_index_ptr,  # [max_request_num + 1, num_buffers_per_req] - tensor mapping req_idx to buffer_idx
    num_reqs,  # number of requests to process
    stride_buffer,  # stride for req_to_buffer_index second dimension
    NUM_BUFFERS_PER_REQ: tl.constexpr,  # number of buffers per request (mtp_step + 1)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for valid indices
    mask = offsets < num_reqs

    # Load request indices
    req_indices = tl.load(req_index_ptr + offsets, mask=mask, other=0)

    # For each request, allocate NUM_BUFFERS_PER_REQ buffers
    for buf_idx in tl.static_range(NUM_BUFFERS_PER_REQ):
        # Load buffer index for this position
        buffer_offset = offsets * NUM_BUFFERS_PER_REQ + buf_idx
        buffer_indices = tl.load(buffer_indexes_ptr + buffer_offset, mask=mask, other=0)

        # Update req_to_buffer_index[req_indices, buf_idx] = buffer_indices
        output_offset = req_indices * stride_buffer + buf_idx
        tl.store(req_to_buffer_index_ptr + output_offset, buffer_indices, mask=mask)


def alloc_buffer_for_req_triton(
    req_index: torch.Tensor,  # [num_reqs] int32/int64 tensor on CUDA
    buffer_indexes: torch.Tensor,  # [num_reqs * (mtp_step + 1)] int32 tensor (can be CPU or CUDA)
    req_to_buffer_index: torch.Tensor,  # [max_request_num + 1, mtp_step + 1] int32 tensor on CUDA
    mtp_step: int = 0,  # number of additional buffers per request (default 0 for non-MTP mode)
):
    num_reqs = req_index.shape[0]
    num_buffers_per_req = mtp_step + 1

    # Ensure inputs are on CUDA
    if not req_index.is_cuda:
        req_index = req_index.cuda()
    if not buffer_indexes.is_cuda:
        buffer_indexes = buffer_indexes.cuda()

    # Ensure correct dtypes
    if req_index.dtype not in [torch.int32, torch.int64]:
        req_index = req_index.to(torch.int32)
    if buffer_indexes.dtype != torch.int32:
        buffer_indexes = buffer_indexes.to(torch.int32)

    # Validate buffer_indexes size
    expected_size = num_reqs * num_buffers_per_req
    assert buffer_indexes.shape[0] == expected_size, (
        f"Expected {expected_size} buffer indices for {num_reqs} requests "
        f"with mtp_step={mtp_step}, but got {buffer_indexes.shape[0]}"
    )

    # Get stride for the second dimension of req_to_buffer_index
    stride_buffer = req_to_buffer_index.stride(0)

    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_reqs, BLOCK_SIZE),)

    alloc_buffer_for_req_kernel[grid](
        req_index,
        buffer_indexes,
        req_to_buffer_index,
        num_reqs,
        stride_buffer,
        NUM_BUFFERS_PER_REQ=num_buffers_per_req,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def get_buffer_index_for_mtp_kernel(
    b_req_idx_ptr,  # [batch_size] - request indices for each batch item
    b_mtp_idx_ptr,  # [batch_size] - MTP indices for each batch item
    req_to_buffer_index_ptr,  # [max_request_num + 1, mtp_step + 1] - tensor mapping req_idx to buffer_idx
    buffer_idx_out_ptr,  # [batch_size] - output buffer indices
    batch_size,
    stride_buffer,  # stride for req_to_buffer_index second dimension
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for valid indices
    mask = offsets < batch_size

    # Load request and MTP indices
    req_indices = tl.load(b_req_idx_ptr + offsets, mask=mask, other=0)
    mtp_indices = tl.load(b_mtp_idx_ptr + offsets, mask=mask, other=0)

    # Calculate offset in req_to_buffer_index
    buffer_offset = req_indices * stride_buffer + mtp_indices

    # Load buffer indices
    buffer_indices = tl.load(req_to_buffer_index_ptr + buffer_offset, mask=mask, other=0)

    # Store output
    tl.store(buffer_idx_out_ptr + offsets, buffer_indices, mask=mask)


def get_buffer_index_for_mtp(
    b_req_idx: torch.Tensor,  # [batch_size] int32 tensor on CUDA
    b_mtp_idx: torch.Tensor,  # [batch_size] int32 tensor on CUDA
    req_to_buffer_index: torch.Tensor,  # [max_request_num + 1, mtp_step + 1] int32 tensor on CUDA
) -> torch.Tensor:
    """
    Get buffer indices for MTP decode mode based on request and MTP indices.

    Args:
        b_req_idx: Request indices for each batch item [batch_size]
        b_mtp_idx: MTP indices for each batch item (0 for main, 1-mtp_step for MTP candidates) [batch_size]
        req_to_buffer_index: Mapping from (req_idx, mtp_idx) to buffer_idx [max_request_num+1, mtp_step+1]

    Returns:
        buffer_idx: Buffer indices for each batch item [batch_size]
    """
    batch_size = b_req_idx.shape[0]

    # Create output tensor
    buffer_idx_out = torch.empty(batch_size, dtype=torch.int32, device="cuda")

    # Get stride for the second dimension of req_to_buffer_index
    stride_buffer = req_to_buffer_index.stride(0)

    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(batch_size, BLOCK_SIZE),)

    get_buffer_index_for_mtp_kernel[grid](
        b_req_idx,
        b_mtp_idx,
        req_to_buffer_index,
        buffer_idx_out,
        batch_size,
        stride_buffer,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return buffer_idx_out
