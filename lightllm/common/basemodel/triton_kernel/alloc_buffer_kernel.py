import torch
import triton
import triton.language as tl


@triton.jit
def alloc_buffer_for_req_kernel(
    req_index_ptr,  # [num_reqs] - indices of requests to allocate buffers for
    buffer_indexes_ptr,  # [num_reqs] - buffer indices to assign (from CPU)
    req_to_buffer_index_ptr,  # [max_request_num + 1] - tensor mapping req_idx to buffer_idx
    num_reqs,  # number of requests to process
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for valid indices
    mask = offsets < num_reqs

    # Load request indices and buffer indices
    req_indices = tl.load(req_index_ptr + offsets, mask=mask, other=0)
    buffer_indices = tl.load(buffer_indexes_ptr + offsets, mask=mask, other=0)

    # Update req_to_buffer_index[req_indices] = buffer_indices
    tl.store(req_to_buffer_index_ptr + req_indices, buffer_indices, mask=mask)


def alloc_buffer_for_req_triton(
    req_index: torch.Tensor,  # [num_reqs] int32/int64 tensor on CUDA
    buffer_indexes: torch.Tensor,  # [num_reqs] int32 tensor (can be CPU or CUDA)
    req_to_buffer_index: torch.Tensor,  # [max_request_num + 1] int32 tensor on CUDA
):
    num_reqs = req_index.shape[0]

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

    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_reqs, BLOCK_SIZE),)

    alloc_buffer_for_req_kernel[grid](
        req_index,
        buffer_indexes,
        req_to_buffer_index,
        num_reqs,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# Convenience function that matches the original API
def alloc_buffer_for_req_wrapper(
    req_manager,
    req_index: list,
    buffer_indexes: torch.Tensor,
):
    """
    Wrapper function to integrate with ReqManagerWithBuffer.

    Usage in ReqManagerWithBuffer:
        def alloc_buffer_for_req(self, req_index: List[int]):
            self.req_has_buffer[req_index] = True
            buffer_indexes = self.mem_manager.alloc_buffer(len(req_index))  # cpu tensor
            # Replace the next line with Triton kernel
            # self.req_to_buffer_index[req_index] = buffer_indexes
            from lightllm.common.basemodel.triton_kernel.alloc_buffer_kernel import alloc_buffer_for_req_triton
            req_index_tensor = torch.tensor(req_index, dtype=torch.int32, device='cuda')
            alloc_buffer_for_req_triton(
                req_index_tensor,
                buffer_indexes,
                self.req_has_buffer,
                self.req_to_buffer_index
            )
    """
    req_index_tensor = torch.tensor(req_index, dtype=torch.int32, device="cuda")
    alloc_buffer_for_req_triton(
        req_index_tensor,
        buffer_indexes,
        req_manager.req_has_buffer,
        req_manager.req_to_buffer_index,
    )
